/*
For more information, please see: http://software.sci.utah.edu

The MIT License

Copyright (c) 2015 Scientific Computing and Imaging Institute,
University of Utah.


Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

#include <Core/Algorithms/Legacy/FiniteElements/BuildMatrix/BuildFEMatrix.h>

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>

#include <Core/Datatypes/Legacy/Field/Mesh.h>
#include <Core/Datatypes/Legacy/Field/VMesh.h>
#include <Core/Datatypes/Legacy/Field/Field.h>
#include <Core/Datatypes/Legacy/Field/VField.h>
#include <Core/GeometryPrimitives/Tensor.h>
#include <Core/Algorithms/Base/AlgorithmPreconditions.h>
#include <Core/Algorithms/Base/AlgorithmVariableNames.h>
#include <Core/Logging/Log.h>

#include <string>
#include <vector>
#include <algorithm>
#include <boost/shared_array.hpp>

using namespace SCIRun;
using namespace SCIRun::Core::Geometry;
using namespace SCIRun::Core::Datatypes;
using namespace SCIRun::Core::Thread;
using namespace SCIRun::Core::Algorithms;
using namespace SCIRun::Core::Algorithms::FiniteElements;
using namespace SCIRun::Core::Logging;


namespace {
// Helper class

class FEMBuilder
{
public:
  FEMBuilder(const AlgorithmBase* algo) :
    algo_(algo), numprocessors_(Parallel::NumCores()),
    barrier_("FEMBuilder Barrier", numprocessors_),
    mesh_(nullptr), field_(nullptr),
    domain_dimension(0), local_dimension_nodes(0),
    local_dimension_add_nodes(0),
    local_dimension_derivatives(0),
    local_dimension(0),
    global_dimension_nodes(0),
    global_dimension_add_nodes(0),
    global_dimension_derivatives(0),
    global_dimension(0), use_tensor_(false), use_scalars_(false)
  {
  }
  
  // Local entry function for none pure function.
  bool build_matrix(FieldHandle input, 
                    DenseMatrixHandle ctable,
                    SparseRowMatrixHandle& output);
  
private:
  const AlgorithmBase* algo_;
  int numprocessors_;
  Barrier barrier_;
  
  VMesh* mesh_;
  VField *field_;
  
  SparseRowMatrixHandle fematrix_;
  
  std::vector<bool> success_;
  
  boost::shared_array<index_type> rows_;
  boost::shared_array<index_type> allcols_;
  std::vector<index_type> colidx_;
  
  index_type domain_dimension;
  
  index_type local_dimension_nodes;
  index_type local_dimension_add_nodes;
  index_type local_dimension_derivatives;
  index_type local_dimension;
  
  index_type global_dimension_nodes;
  index_type global_dimension_add_nodes;
  index_type global_dimension_derivatives;
  index_type global_dimension; 
  
  // A copy of the tensors list that was generated by SetConductivities
  bool use_tensor_;
  std::vector<std::pair<std::string, Tensor> > tensors_;
  
  bool use_scalars_;
  std::vector<std::pair<std::string, double> > scalars_;
  
  // Entry point for the parallel version
  void parallel(int proc);
  
private:
  
  inline void add_lcl_gbl(index_type row, const std::vector<index_type> &cols, const std::vector<double> &lcl_a)
  {
    for (size_t i = 0; i < lcl_a.size(); i++)
      fematrix_->coeffRef(row, cols[i]) += lcl_a[i];
  }
  
private:
  
  void create_numerical_integration(std::vector<VMesh::coords_type > &p,
                                    std::vector<double> &w,
                                    std::vector<std::vector<double> > &d);
  bool build_local_matrix(VMesh::Elem::index_type c_ind,
                          index_type row, 
                          std::vector<double> &l_stiff,
                          std::vector<VMesh::coords_type> &p,
                          std::vector<double> &w,
                          std::vector<std::vector<double> >  &d);
  bool build_local_matrix_regular(VMesh::Elem::index_type c_ind,
                                  index_type row, 
                                  std::vector<double> &l_stiff,
                                  std::vector<VMesh::coords_type> &p,
                                  std::vector<double> &w,
                                  std::vector<std::vector<double> >  &d, 
                                  std::vector<std::vector<double> > &precompute);
  bool setup();
  
};


bool
FEMBuilder::build_matrix(FieldHandle input, 
                         DenseMatrixHandle ctable,
                         SparseRowMatrixHandle& output)
{
  // Get virtual interface to data
  field_ = input->vfield();
  mesh_  = input->vmesh();
  
#ifdef SCIRUN4_CODE_TO_BE_ENABLED_LATER
  // If we have the Conductivity property use it, if not we assume the values on
  // the data to be the actual tensors.
  field_->get_property("conductivity_table",tensors_);
#endif

  // We added a second system of adding a conductivity table, using a matrix
  // Convert that matrix into the conductivity table
  if (ctable)
  {
    tensors_.clear();
    auto mat = ctable;
    // Only if we can convert it into a dense matrix, otherwise skip it
    if (mat)
    {
      auto data = mat->data();
      size_type m = mat->nrows();
      size_type n = mat->ncols();
      Tensor tensor; 
      
      // Case the table has isotropic conductivities
      if (mat->ncols() == 1)
      {
        for (size_type p=0; p<m;p++)
        {
          // Set the diagonals to the proper version.
          tensor.val(0,0) = data[p*n+0];
          tensor.val(1,0) = 0.0;
          tensor.val(2,0) = 0.0;
          tensor.val(0,1) = 0.0;
          tensor.val(1,1) = data[p*n+0];
          tensor.val(2,1) = 0.0;
          tensor.val(0,2) = 0.0;
          tensor.val(1,2) = 0.0;
          tensor.val(2,2) = data[p*n+0];
          tensors_.push_back(std::make_pair("",tensor));
        }
      }
      
      // Use our compressed way of storing tensors 
      if (mat->ncols() == 6)
      {
        for (size_type p=0; p<m;p++)
        {
          tensor.val(0,0) = data[0+p*n];
          tensor.val(1,0) = data[1+p*n];
          tensor.val(2,0) = data[2+p*n];
          tensor.val(0,1) = data[1+p*n];
          tensor.val(1,1) = data[3+p*n];
          tensor.val(2,1) = data[4+p*n];
          tensor.val(0,2) = data[2+p*n];
          tensor.val(1,2) = data[4+p*n];
          tensor.val(2,2) = data[5+p*n];
          tensors_.push_back(std::make_pair("",tensor));
        }
      }
      
      // Use the full symmetric tensor. We will make the tensor symmetric here.
      if (mat->ncols() == 9)
      {
        for (size_type p=0; p<m;p++)
        {
          tensor.val(0,0) = data[0+p*n];
          tensor.val(1,0) = data[1+p*n];
          tensor.val(2,0) = data[2+p*n];
          tensor.val(0,1) = data[1+p*n];
          tensor.val(1,1) = data[4+p*n];
          tensor.val(2,1) = data[5+p*n];
          tensor.val(0,2) = data[2+p*n];
          tensor.val(1,2) = data[5+p*n];
          tensor.val(2,2) = data[8+p*n];
          tensors_.push_back(std::make_pair("",tensor));
        }
      }
    }
  }
  
  success_.resize(numprocessors_,true);
  
  // Start the multi threaded FE matrix builder.
  Parallel::RunTasks([this](int i) { parallel(i); }, numprocessors_);
  for (size_t j=0; j<success_.size(); j++)
  {
    if (!success_[j])
    {
      std::ostringstream oss;
      oss << "Algorithm failed in thread " << j;
      algo_->error(oss.str());
      return false;
    }
  }
  
  // Make sure it is symmetric
  if (algo_->get(BuildFEMatrixAlgo::ForceSymmetry).toBool())
  {
    // Make sure the matrix is fully symmetric, this compensates for round off
    // errors
    SparseRowMatrix transpose = fematrix_->transpose();
    output.reset(new SparseRowMatrix(0.5*(transpose + *fematrix_)));
  }
  else
  {
    // Assume that the builder did a good job and the matrix is numerically almost
    // symmetric
    output = fematrix_;
  }
  
  return true;
}


void 
FEMBuilder::create_numerical_integration(std::vector<VMesh::coords_type> &p,
                                         std::vector<double> &w,
                                         std::vector<std::vector<double> > &d)
{
  //ScopedTimeLogger s1("FEMBuilder::create_numerical_integration");
  int int_basis = 1;
  if (mesh_->is_quad_element() || 
      mesh_->is_hex_element() || 
      mesh_->is_prism_element())
  {
    int_basis = 2;
  }
  
  mesh_->get_gaussian_scheme(p,w,int_basis);
  d.resize(p.size());
  for (size_t j=0; j<p.size();j++)
  {
    mesh_->get_derivate_weights(p[j],d[j],1);
    size_t pad_size = ( 3 - p[ j ].size() ) * d[ j ].size();
    
    if (pad_size > 0)
      d[j].resize(pad_size + d[j].size(), 0.0);
  }
}


/// build line of the local stiffness matrix

bool
FEMBuilder::build_local_matrix(VMesh::Elem::index_type c_ind,
                               index_type row,
                               std::vector<double> &l_stiff,
                               std::vector<VMesh::coords_type > &p,
                               std::vector<double> &w,
                               std::vector<std::vector<double> >  &d)
{
  //ScopedTimeLogger s0("FEMBuilder::build_local_matrix");
  Tensor T;
  
  if (tensors_.empty())
  {
    field_->get_value(T,c_ind);
  }
  else
  {
    int tensor_index;
    field_->get_value(tensor_index,c_ind);
    T = tensors_[tensor_index].second;
  }
  
  auto Ca = T.val(0,0);
  auto Cb = T.val(0,1);
  auto Cc = T.val(0,2);
  auto Cd = T.val(1,1);
  auto Ce = T.val(1,2);
  auto Cf = T.val(2,2);
  
  if ( (Ca==0) && (Cb==0) && (Cc==0) && (Cd==0) && (Ce==0) && (Cf==0) )
  {
    for (int j = 0; j<local_dimension; j++)
    {
      l_stiff[j] = 0.0;
    }
  }
  else
  {
    /// @todo: replace with std::fill
    for (int i=0; i<local_dimension; i++)
      l_stiff[i] = 0.0;
    
    auto local_dimension2=2*local_dimension;
    
    // These calls are direct lookups in the base of the VMesh
    // The compiler should optimize these well
    auto vol = mesh_->get_element_size();
    const int dim = mesh_->dimensionality();
    
    if (dim < 1 || dim > 3)
    {
      algo_->error("Mesh dimension is 0 or larger than 3, for which no FE implementation is available");
      return false;    
    }
    for (size_t i = 0; i < d.size(); i++)
    {
      double Ji[9];
      // Call to virtual interface, this should be one internal call
      auto detJ = mesh_->inverse_jacobian(p[i],c_ind,Ji);   
      
      // If Jacobian is negative there is a problem with the mesh
      if (detJ <= 0.0)
      {
        algo_->error("Mesh has elements with negative jacobians, check the order of the nodes that define an element");
        return false;
      }
      
      // Volume associated with the local Gaussian Quadrature point:
      // weightfactor * Volume Unit element * Volume ratio (real element/unit element)
      detJ *= w[i] * vol;
      
      // Build local stiffness matrix
      // Get the local derivatives of the basis functions in the basis element
      // They are all the same and are thus precomputed in matrix d
      const double *Nxi = &d[i][0];
      const double *Nyi = &d[i][local_dimension];
      const double *Nzi = &d[i][local_dimension2];
      // Gradients associated with the node we are calculating
      const auto& Nxip = Nxi[row];
      const auto &Nyip = Nyi[row];
      const auto &Nzip = Nzi[row];
      // Calculating gradient shape function * inverse Jacobian * volume scaling factor
      const auto uxp = detJ*(Nxip*Ji[0] + Nyip*Ji[1] + Nzip*Ji[2]);
      const auto uyp = detJ*(Nxip*Ji[3] + Nyip*Ji[4] + Nzip*Ji[5]);
      const auto uzp = detJ*(Nxip*Ji[6] + Nyip*Ji[7] + Nzip*Ji[8]);
      // Matrix multiplication with conductivity tensor :
      const auto uxyzpabc = uxp*Ca + uyp*Cb + uzp*Cc;
      const auto uxyzpbde = uxp*Cb + uyp*Cd + uzp*Ce;
      const auto uxyzpcef = uxp*Cc + uyp*Ce + uzp*Cf;
      
      // The above is constant for this node. Now multiply with the weight function
      // We assume the weight factors are the same as the local gradients 
      // Galerkin approximation:
      
      for (int j = 0; j<local_dimension; j++)
      {
        const auto &Nxj = Nxi[j];
        const auto &Nyj = Nyi[j];
        const auto &Nzj = Nzi[j];
        
        // Matrix multiplication Gradient with inverse Jacobian:
        const auto ux = Nxj*Ji[0] + Nyj*Ji[1] + Nzj*Ji[2];
        const auto uy = Nxj*Ji[3] + Nyj*Ji[4] + Nzj*Ji[5];
        const auto uz = Nxj*Ji[6] + Nyj*Ji[7] + Nzj*Ji[8];
        
        // Add everything together into one coefficient of the matrix
        l_stiff[j] += ux*uxyzpabc+uy*uxyzpbde+uz*uxyzpcef;
      }
    }
  }
  
  return true;
}


bool 
FEMBuilder::build_local_matrix_regular(VMesh::Elem::index_type c_ind,
                                       index_type row,
                                       std::vector<double> &l_stiff,
                                       std::vector<VMesh::coords_type> &p,
                                       std::vector<double> &w,
                                       std::vector<std::vector<double> >  &d,
                                       std::vector<std::vector<double> > &precompute)
{
  //ScopedTimeLogger s0("FEMBuilder::build_local_matrix_regular");
  Tensor T;
  
  if (tensors_.empty())
  {
    // Call to virtual interface. Get the tensor value. Actually this call relies
    // on the automatic casting feature of the virtual interface to convert scalar
    // values into a tensor.
    field_->get_value(T,c_ind);
  }
  else
  {
    int tensor_index;
    field_->get_value(tensor_index,c_ind);
    T = tensors_[tensor_index].second;
  }
  
  auto Ca = T.val(0,0);
  auto Cb = T.val(0,1);
  auto Cc = T.val(0,2);
  auto Cd = T.val(1,1);
  auto Ce = T.val(1,2);
  auto Cf = T.val(2,2);
  
  if ( (Ca==0) && (Cb==0) && (Cc==0) && (Cd==0) && (Ce==0) && (Cf==0) )
  {
    for (int j = 0; j<local_dimension; j++)
    {
      l_stiff[j] = 0.0;
    }
  }
  else
  {
    
    if (precompute.empty())
    {
      precompute.resize(d.size());
      for (int m=0; m < static_cast<int>(d.size()); m++)
      {
        precompute[m].resize(10);
      }
      
      for(int i=0; i<local_dimension; i++)
        l_stiff[i] = 0.0;
      
      auto local_dimension2=2*local_dimension;

      auto vol = mesh_->get_element_size();
      
      for (size_t i = 0; i < d.size(); i++)
      {
        auto& pc = precompute[i];
        
        double Ji[9];
        auto detJ = mesh_->inverse_jacobian(p[i], c_ind, Ji);
        
        // Volume elements can return negative determinants if the order of elements
        // is put in a different order
        /// @todo: It seems to be that a negative determinant is not necessarily bad, 
        // we should be more flexible on this point
        if (detJ <= 0.0) 
        {
          algo_->error("Mesh has elements with negative jacobians, check the order of the nodes that define an element");
          return false;
        }
        // Volume associated with the local Gaussian Quadrature point:
        // weightfactor * Volume Unit element * Volume ratio (real element/unit element)
        detJ*=w[i]*vol;
        
        pc[0] = Ji[0];
        pc[1] = Ji[1];
        pc[2] = Ji[2];
        pc[3] = Ji[3];
        pc[4] = Ji[4];
        pc[5] = Ji[5];
        pc[6] = Ji[6];
        pc[7] = Ji[7];
        pc[8] = Ji[8];
        pc[9] = detJ;
        
        // Build local stiffness matrix
        // Get the local derivatives of the basis functions in the basis element
        // They are all the same and are thus precomputed in matrix d
        const double *Nxi = &d[i][0];
        const double *Nyi = &d[i][local_dimension];
        const double *Nzi = &d[i][local_dimension2];
        // Gradients associated with the node we are calculating
        const auto &Nxip = Nxi[row];
        const auto &Nyip = Nyi[row];
        const auto &Nzip = Nzi[row];
        // Calculating gradient shape function * inverse Jacobian * volume scaling factor
        const auto uxp = pc[9]*(Nxip*pc[0]+Nyip*pc[1]+Nzip*pc[2]);
        const auto uyp = pc[9]*(Nxip*pc[3]+Nyip*pc[4]+Nzip*pc[5]);
        const auto uzp = pc[9]*(Nxip*pc[6]+Nyip*pc[7]+Nzip*pc[8]);
        // Matrix multiplication with conductivity tensor :
        const auto uxyzpabc = uxp*Ca+uyp*Cb+uzp*Cc;
        const auto uxyzpbde = uxp*Cb+uyp*Cd+uzp*Ce;
        const auto uxyzpcef = uxp*Cc+uyp*Ce+uzp*Cf;
        
        // The above is constant for this node. Now multiply with the weight function
        // We assume the weight factors are the same as the local gradients 
        // Galerkin approximation:
        
        for (int j = 0; j<local_dimension; j++)
        {
          const auto &Nxj = Nxi[j];
          const auto &Nyj = Nyi[j];
          const auto &Nzj = Nzi[j];
          
          // Matrix multiplication Gradient with inverse Jacobian:
          const auto ux = Nxj*pc[0]+Nyj*pc[1]+Nzj*pc[2];
          const auto uy = Nxj*pc[3]+Nyj*pc[4]+Nzj*pc[5];
          const auto uz = Nxj*pc[6]+Nyj*pc[7]+Nzj*pc[8];
          
          // Add everything together into one coefficient of the matrix
          l_stiff[j] += ux*uxyzpabc+uy*uxyzpbde+uz*uxyzpcef;
        }
      }
    }
    else
    {      
      for(int i=0; i<local_dimension; i++)
        l_stiff[i] = 0.0;
      
      auto local_dimension2=2*local_dimension;
      
      for (size_t i = 0; i < d.size(); i++)
      {
        std::vector<double>& pc = precompute[i];
        
        // Build local stiffness matrix
        // Get the local derivatives of the basis functions in the basis element
        // They are all the same and are thus precomputed in matrix d
        const double *Nxi = &d[i][0];
        const double *Nyi = &d[i][local_dimension];
        const double *Nzi = &d[i][local_dimension2];
        // Gradients associated with the node we are calculating
        const auto &Nxip = Nxi[row];
        const auto &Nyip = Nyi[row];
        const auto &Nzip = Nzi[row];
        // Calculating gradient shape function * inverse Jacobian * volume scaling factor
        const auto uxp = pc[9]*(Nxip*pc[0]+Nyip*pc[1]+Nzip*pc[2]);
        const auto uyp = pc[9]*(Nxip*pc[3]+Nyip*pc[4]+Nzip*pc[5]);
        const auto uzp = pc[9]*(Nxip*pc[6]+Nyip*pc[7]+Nzip*pc[8]);
        // Matrix multiplication with conductivity tensor :
        const auto uxyzpabc = uxp*Ca+uyp*Cb+uzp*Cc;
        const auto uxyzpbde = uxp*Cb+uyp*Cd+uzp*Ce;
        const auto uxyzpcef = uxp*Cc+uyp*Ce+uzp*Cf;
        
        // The above is constant for this node. Now multiply with the weight function
        // We assume the weight factors are the same as the local gradients 
        // Galerkin approximation:
        
        for (int j = 0; j<local_dimension; j++)
        {
          const auto &Nxj = Nxi[j];
          const auto &Nyj = Nyi[j];
          const auto &Nzj = Nzi[j];
          
          // Matrix multiplication Gradient with inverse Jacobian:
          const auto ux = Nxj*pc[0]+Nyj*pc[1]+Nzj*pc[2];
          const auto uy = Nxj*pc[3]+Nyj*pc[4]+Nzj*pc[5];
          const auto uz = Nxj*pc[6]+Nyj*pc[7]+Nzj*pc[8];
          
          // Add everything together into one coefficient of the matrix
          l_stiff[j] += ux*uxyzpabc+uy*uxyzpbde+uz*uxyzpcef;
        }
      }
    }
  }
  
  return true;
}


bool
FEMBuilder::setup()
{	
  // The domain dimension
  domain_dimension = mesh_->dimensionality();
  if (domain_dimension < 1) 
  {
    algo_->error("This mesh type cannot be used for FE computations");
    return false;
  }
  
  local_dimension_nodes = mesh_->num_nodes_per_elem();
  if (field_->basis_order() == 2)
  {
    local_dimension_add_nodes = mesh_->num_enodes_per_elem();
  }
  else
  {
    local_dimension_add_nodes = 0;
  }
  
  local_dimension_derivatives = 0;
  
  // Local degrees of freedom per element
  local_dimension = local_dimension_nodes + 
  local_dimension_add_nodes + 
  local_dimension_derivatives; ///< degrees of freedom (dofs) of system
  
  VMesh::Node::size_type mns;
  mesh_->size(mns);
  
  // Number of mesh points (not necessarily number of nodes)
  global_dimension_nodes = mns;
  if (field_->basis_order() == 2) // quadratic
  {
    mesh_->synchronize(Mesh::ENODES_E);
    global_dimension_add_nodes = mesh_->num_enodes();
  }
  else
  {
    global_dimension_add_nodes = 0;
  }
  
  global_dimension_derivatives = 0;
  global_dimension = global_dimension_nodes+
  global_dimension_add_nodes+
  global_dimension_derivatives;
  
  if (mns > 0) 
  {
    // We only need edges for the higher order basis in case of quartic Lagrangian
    // Hence we should only synchronize it for this case
    if (global_dimension_add_nodes > 0) 
      mesh_->synchronize(Mesh::EDGES_E|Mesh::NODE_NEIGHBORS_E);
    else
      mesh_->synchronize(Mesh::NODE_NEIGHBORS_E);
  }
  else
  {
    algo_->error("Mesh size < 0");
    success_[0] = false;
  }
  Log::get() << DEBUG_LOG << "Allocating buffer for nonzero row indices of size: " << (global_dimension+1);
  rows_.reset(new index_type[global_dimension+1]);
  
  colidx_.resize(numprocessors_+1);
  return true;
}



// -- callback routine to execute in parallel
void 
FEMBuilder::parallel(int proc_num)
{
  success_[proc_num] = true;
  
  if (proc_num == 0)
  {
    try
    {
      success_[proc_num] = setup();
    }
    catch (...)
    {
      algo_->error("BuildFEMatrix could not setup FE Stiffness computation");
      success_[proc_num] = false;
    }
  }
  
  barrier_.wait();
  
  // In case one of the threads fails, we should have them fail all
  for (int q = 0; q < numprocessors_; q++)
  {
    if (!success_[q])
    {
      std::ostringstream oss;
      oss << "FEMBuilder::setup failed in thread " << q;
      algo_->error(oss.str());
      return;
    }
  }
  
  /// distributing dofs among processors
  const index_type start_gd = (global_dimension * proc_num)/numprocessors_;
  const index_type end_gd  = (global_dimension * (proc_num+1))/numprocessors_;
  
  /// creating sparse matrix structure
  std::vector<index_type> mycols;
  
  VMesh::Elem::array_type ca;
  VMesh::Node::array_type na;
  VMesh::Edge::array_type ea;
  std::vector<index_type> neib_dofs;
  
  /// loop over system dofs for this thread
  int cnt = 0;
  size_type size_gd = end_gd-start_gd;
  auto updateFrequency = 2*size_gd / 100;
  try
  {
    mycols.reserve((end_gd - start_gd)*local_dimension*8);  //<! rough estimate
    
    for (VMesh::Node::index_type i = start_gd; i<end_gd; ++i)
    {
      rows_[i] = mycols.size();
      
      neib_dofs.clear();
      /// check for nodes
      if (i < global_dimension_nodes)
      {
        /// get neighboring cells for node
        mesh_->get_elems(ca, i);
      }
      else if (i < global_dimension_nodes+global_dimension_add_nodes)
      {
        /// check for additional nodes at edges
        /// get neighboring cells for node
        VMesh::Edge::index_type ii(i-global_dimension_nodes);
        mesh_->get_elems(ca,ii);
      }
      else
      {
        // There is some functionality implemented for higher order basis functions,
        // but it seems not to be accessible, entirely implemented nor validated.
        algo_->warning("BuildFEMatrix only supports linear basis functions.");
      }
      
      for(size_t j = 0; j < ca.size(); j++)
      {
        /// get neighboring nodes
        mesh_->get_nodes(na, ca[j]);
        
        for(size_t k = 0; k < na.size(); k++) 
        {
          neib_dofs.push_back(static_cast<index_type>(na[k]));
        }
        
        /// check for additional nodes at edges
        if (global_dimension_add_nodes)
        {
          /// get neighboring edges
          mesh_->get_edges(ea, ca[j]);
          
          for(size_t k = 0; k < ea.size(); k++)
            neib_dofs.push_back(global_dimension + ea[k]);
        }
      }
      
      std::sort(neib_dofs.begin(), neib_dofs.end());
      
      for (size_t j=0; j<neib_dofs.size(); j++)
      {
        if (j == 0 || neib_dofs[j] != mycols.back())
        {
          mycols.push_back(neib_dofs[j]);
        }
      }
      if (proc_num == 0) 
      {
        cnt++;
        if (cnt == updateFrequency)
        {
          cnt = 0;
          algo_->update_progress_max(i,2*size_gd);
        }
      }    
    }
    
    colidx_[proc_num] = mycols.size();
    success_[proc_num] = true;
  }
  catch (...)
  {
    algo_->error("BuildFEMatrix crashed mapping out stiffness matrix");
    success_[proc_num] = false;
  }
  
  /// check point
  barrier_.wait();
  
  // Bail out if one of the processes failed
  for (int q=0; q<numprocessors_;q++)
  {
    if (!success_[q])
    {
      return;
    }
  }
  
  std::vector<std::vector<double> > precompute;		
  index_type st = 0;
  
  if (proc_num == 0)
    allcols_.reset();
  
  try
  {
    if (proc_num == 0)
    {
      for(int i=0; i<numprocessors_; i++)
      {
        const index_type ns = colidx_[i];
        colidx_[i] = st;
        st += ns;
      }
      
      colidx_[numprocessors_] = st;
      allcols_.reset(new index_type[st]);
    }
    success_[proc_num] = true;
  }
  catch (...)
  {
    if (proc_num == 0)
      allcols_.reset();
    
    algo_->error("Could not allocate enough memory");
    success_[proc_num] = false;
  }	
  
  /// check point
  barrier_.wait();
  
  // Bail out if one of the processes failed
  for (int q=0; q<numprocessors_;q++)
  {
    if (! success_[q])
      return;
  }
  
  try
  {
    /// updating global column by each of the processors
    const index_type s = colidx_[proc_num];
    const size_t n = mycols.size();
    
    for(size_t i=0; i<n; i++)
      allcols_[i+s] = mycols[i];
    
    for(index_type i = start_gd; i<end_gd; i++)
      rows_[i] += s;
    
    success_[proc_num] = true;
  }
  catch (...)
  {
    algo_->error("BuildFEMatrix crashed while setting up row compression");
    success_[proc_num] = false;
  }	
  
  
  /// check point
  barrier_.wait();
  
  // Bail out if one of the processes failed
  for (auto q=0; q<numprocessors_; q++)
  {
    if (!success_[q])
      return;
  }
  
  try
  {	
    /// the main thread makes the matrix
    if (proc_num == 0)
    {
      rows_[global_dimension] = st;
      algo_->remark("Creating fematrix on main thread.");
      fematrix_ = boost::make_shared<SparseRowMatrix>(global_dimension, global_dimension, rows_.get(), allcols_.get(), st);
      rows_.reset();
      allcols_.reset();
    }
    success_[proc_num] = true;
  }
  catch (...)
  {
    algo_->error("BuildFEMatrix crashed while creating final stiffness matrix");
    success_[proc_num] = false;
  }	
  
  /// check point
  barrier_.wait();
  
  // Bail out if one of the processes failed
  for (auto q=0; q<numprocessors_;q++)
  {
    if (!success_[q])
      return;
  }
  
  try
  {
    /// zeroing in parallel
    const auto ns = colidx_[proc_num];
    const auto ne = colidx_[proc_num+1];
    auto a = &(fematrix_->valuePtr()[ns]), ae=&(fematrix_->valuePtr()[ne]);
    while (a<ae) *a++=0.0;
    
    std::vector<VMesh::coords_type > ni_points;
    std::vector<double> ni_weights;
    std::vector<std::vector<double> > ni_derivatives;
    
    create_numerical_integration(ni_points, ni_weights, ni_derivatives);
    
    std::vector<double> lsml; ///< line of local stiffnes matrix
    lsml.resize(local_dimension);
    
    /// loop over system dofs for this thread
    cnt = 0;
    size_gd = end_gd-start_gd;
    for (VMesh::Node::index_type i = start_gd; i<end_gd; ++i)
    {
      if (i < global_dimension_nodes)
      {
        /// check for nodes
        /// get neighboring cells for node
        mesh_->get_elems(ca,i);
      }
      else if (i < global_dimension_nodes + global_dimension_add_nodes)
      {
        /// check for additional nodes at edges
        /// get neighboring cells for additional nodes
        VMesh::Edge::index_type ii(i-global_dimension_nodes);
        mesh_->get_elems(ca,ii);
      }
      else
      {
        // There is some functionality implemented for higher order basis functions,
        // but it seems not to be accessible, entirely implemented nor validated.
        algo_->warning("BuildFEMatrix only supports linear basis functions.");
      }
      
      /// loop over elements attributed elements
      
      if (mesh_->is_regularmesh())
      {
        for (size_t j = 0; j < ca.size(); j++)
        {
          mesh_->get_nodes(na, ca[j]); ///< get neighboring nodes
          neib_dofs.resize(na.size());
          for(size_t k = 0; k < na.size(); k++)
          {
            neib_dofs[k] = na[k]; // Must cast to (int) for SGI compiler :-(
          }
          
          for(size_t k = 0; k < na.size(); k++)
          {
            if (na[k] == i) 
            {
              build_local_matrix_regular(ca[j], k , lsml, ni_points, ni_weights, ni_derivatives,precompute);
              add_lcl_gbl(i, neib_dofs, lsml);
            }
          }
        }
      }
      else
      {
        for (size_t j = 0; j < ca.size(); j++)
        {
          neib_dofs.clear();
          mesh_->get_nodes(na, ca[j]); ///< get neighboring nodes
          for(size_t k = 0; k < na.size(); k++)
          {
            neib_dofs.push_back(na[k]); // Must cast to (int) for SGI compiler :-(
          }
          /// check for additional nodes at edges
          if (global_dimension_add_nodes)
          {
            mesh_->get_edges(ea, ca[j]); ///< get neighboring edges
            for(size_t k = 0; k < ea.size(); k++)
            {
              neib_dofs.push_back(global_dimension + ea[k]);
            }
          }
          
          ASSERT(static_cast<int>(neib_dofs.size()) == local_dimension);
          
          for(size_t k = 0; k < na.size(); k++)
          {
            if (na[k] == i) 
            {
              build_local_matrix(ca[j], k , lsml, ni_points, ni_weights, ni_derivatives);
              add_lcl_gbl(i, neib_dofs, lsml);
            }
          }
          
          if (global_dimension_add_nodes)
          {
            for (size_t k = 0; k < ea.size(); k++)
            {
              if (global_dimension + static_cast<int>(ea[k]) == i)
              {
                build_local_matrix(ca[j], k+na.size(), lsml, ni_points, ni_weights, ni_derivatives);
                add_lcl_gbl(i, neib_dofs, lsml);
              }
            }
          }
        }
      }
      
      if (proc_num == 0) 
      {
        cnt++;
        if (cnt == updateFrequency)
        {
          cnt = 0;
          Log::get() << DEBUG_LOG << "Updating progress 2 to: " << i+size_gd << " / " << 2*size_gd;
          algo_->update_progress_max(i+size_gd,2*size_gd);
        }
      }
    }
    success_[proc_num] = true;
  }
  catch (...)
  {
    algo_->error("BuildFEMatrix crashed while filling out stiffness matrix");
    success_[proc_num] = false;
  }	
  
  barrier_.wait();
  
  // Bail out if one of the processes failed
  for (int q=0; q<numprocessors_; q++)
  {
    if (!success_[q])
      return;
  }
}
}

const AlgorithmParameterName BuildFEMatrixAlgo::ForceSymmetry("ForceSymmetry");
const AlgorithmParameterName BuildFEMatrixAlgo::GenerateBasis("GenerateBasis");

bool 
BuildFEMatrixAlgo::run(FieldHandle input, DenseMatrixHandle ctable, SparseRowMatrixHandle& output) const
{
  ScopedAlgorithmStatusReporter s(this, "BuildFEMatrix");
  
  if (!input)
  {
    error("Could not obtain input field");
    return false;
  }
  
  if (input->vfield()->is_vector())
  {
    error("This function has not yet been defined for elements with vector data");
    return false;
  }
  
  if (input->vfield()->basis_order()!=0)
  {
    error("This function has only been defined for data that is located at the elements");
    return false;
  }
  
  if (ctable)
  {
    if ((ctable->ncols() != 1)&&(ctable->ncols() != 6)&&(ctable->ncols() != 9))
    {
      error("Conductivity table needs to have 1, 6, or 9 columns");
      return false;
    } 
    if (ctable->nrows() == 0)
    { 
      error("ConductivityTable is empty");
      return false;
    }
  }
  
  FEMBuilder builder(this);
  
  if (get(GenerateBasis).toBool())
  {
    if (!ctable)
    {
      std::vector<std::pair<std::string,Tensor> > tens;
      
      input->properties().get_property("conductivity_table",tens);
      
      if (!tens.empty())
      {
        ctable.reset(new DenseMatrix(tens.size(), 1));
        auto data = ctable->data();
        for (size_t i=0; i<tens.size();i++)
        {
          auto t = tens[i].second.val(0,0);
          data[i] = t;
        }
      }
    }
    
    if (ctable)
    {
      auto nconds = ctable->nrows();
      if ( (input->vmesh()->generation() != generation_) ||
          (!basis_fematrix_) )
      {
        auto con = boost::make_shared<DenseMatrix>(nconds, 1, 0.0);
        auto data = con->data();
        
        if (! builder.build_matrix(input, con, basis_fematrix_) )
        {
          error("Build matrix method failed when building FEMatrix structure");
          return false;
        }
        
        if (!basis_fematrix_)
        {
          error("Failed to build FEMatrix structure");
          return false;
        }
        
        basis_values_.resize(nconds);
        for (size_type i=0; i < nconds; i++)
        {
          SparseRowMatrixHandle stiffness;
          /// @todo: can initialize array using std::fill
          data[i] = 1.0;
          
          if (! builder.build_matrix(input, con, stiffness) )
          {
            error("Build matrix method failed for one of the tissue types");
            return false;
          }
          
          if (!stiffness)
          {
            error("Failed to build FEMatrix component for one of the tissue types");
            return false;
          }
          
          basis_values_[i].resize(stiffness->nonZeros());
          for (size_type p=0; p< stiffness->nonZeros(); p++)
          {
            basis_values_[i][p] = stiffness->valuePtr()[p];
          }
          data[i] = 0.0;
        }
        
        generation_ = input->vmesh()->generation();
      }
      
      output.reset(basis_fematrix_->clone());

      auto sum = output->valuePtr();
      auto cdata = ctable->data();
      auto n = ctable->ncols();
      
      if (!basis_values_.empty())
      {
        for (size_t p=0; p < basis_values_[0].size(); p++)
          sum[p] = 0.0;
      }
      
      for (auto i=0; i<nconds; i++)
      {
        auto weight = cdata[i*n];
        for (size_t p=0; p < basis_values_[i].size(); p++)
        {
          sum[p] += weight * basis_values_[i][p];
        }
      }
      
    }
    else
    {
      error("No conductivity table present: The generate_basis option only works for indexed conductivities");
      return false;
    }
  }
  
  if (! builder.build_matrix(input,ctable,output) )
  {
    error("Build matrix method failed to build output matrix");
    return false;
  }
  
  if (!output)
  {    
    error("Could not build output matrix");
    return false;
  }
  
  return true;
}

const AlgorithmInputName BuildFEMatrixAlgo::Conductivity_Table("Conductivity_Table");
const AlgorithmOutputName BuildFEMatrixAlgo::Stiffness_Matrix("Stiffness_Matrix");

AlgorithmOutput BuildFEMatrixAlgo::run(const AlgorithmInput& input) const
{
  auto field = input.get<Field>(Variables::InputField);
  auto ctable = input.get<DenseMatrix>(Conductivity_Table);

  SparseRowMatrixHandle stiffness;
  if (!run(field, ctable, stiffness))
    THROW_ALGORITHM_PROCESSING_ERROR("False returned on legacy run call.");

  AlgorithmOutput output;
  output[Stiffness_Matrix] = stiffness;
  return output;
}
