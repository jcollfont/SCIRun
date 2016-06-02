/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2009 Scientific Computing and Imaging Institute,
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


// Testing libraries
#include <Testing/ModuleTestBase/ModuleTestBase.h>
#include <Testing/Utils/MatrixTestUtilities.h>
#include <Testing/Utils/SCIRunUnitTests.h>

// General Libraries
#include <Core/Algorithms/Base/AlgorithmPreconditions.h>

// DataType libraries
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColumnMatrix.h>
#include <Core/Datatypes/MatrixTypeConversions.h>

// Tikhonov specific
#include <Modules/Legacy/Inverse/SolveInverseProblemWithTikhonovImpl.h>
#include <Modules/Legacy/Inverse/SolveInverseProblemWithTikhonov.h>

using namespace SCIRun;
using namespace SCIRun::Testing;
using namespace SCIRun::Modules;
//  using namespace SCIRun::Modules::Math;
using namespace SCIRun::Core::Datatypes;
using namespace SCIRun::Core::Algorithms;
using namespace SCIRun::Dataflow::Networks;
using namespace SCIRun::TestUtils;
using namespace SCIRun::Modules::Inverse;
using namespace SCIRun::Core::Algorithms::Inverse;
using ::testing::_;
using ::testing::NiceMock;
using ::testing::DefaultValue;
using ::testing::Return;
using ::testing::Values;
using ::testing::Combine;
using ::testing::Range;


class TikhonovFunctionalTest : public ModuleTest
{
protected:
    UseRealModuleStateFactory f;
};

namespace  {
    const double abs_error = 1e-6;
}

/// -------- INPUTS TESTS ------------ ///

// NULL fwd matrix + NULL measure data
TEST_F(TikhonovFunctionalTest, loadNullFwdMatrixANDNullData)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle nullMatrix, nullColumnMatrix;
  // input data
  stubPortNWithThisData(tikAlgImp, 0, nullMatrix);
  stubPortNWithThisData(tikAlgImp, 2, nullColumnMatrix);
  // check result
  EXPECT_THROW(tikAlgImp->execute(), NullHandleOnPortException);

}

// ID fwd matrix + null measured data
TEST_F(TikhonovFunctionalTest, loadIDFwdMatrixANDNullData)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
  MatrixHandle nullColumnMatrix;              // measurement data (null)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, nullColumnMatrix);
  // check result
  EXPECT_THROW(tikAlgImp->execute(), NullHandleOnPortException);

}

// NULL fwd matrix + RANF measured data
TEST_F(TikhonovFunctionalTest, loadNullFwdMatrixANDRandData)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix;    // forward matrix (IDentityt)
  MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));    // measurement data (rand)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, measuredData);
  // check result
  EXPECT_THROW(tikAlgImp->execute(), NullHandleOnPortException);

}

// ID squared fwd matrix + RAND measured data
TEST_F(TikhonovFunctionalTest, loadIDFwdMatrixANDRandData)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
  MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, measuredData);
  // check result
  EXPECT_NO_THROW(tikAlgImp->execute());

}

// ID non-square fwd matrix + RAND measured data  (underdetermined)
TEST_F(TikhonovFunctionalTest, loadIDNonSquareFwdMatrixANDRandData)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 4)));    // forward matrix (IDentityt)
  MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, measuredData);
  // check result
  EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID non-square fwd matrix + RAND measured data  (overdetermined)
TEST_F(TikhonovFunctionalTest, loadIDNonSquareFwdMatrixANDRandData2)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 3)));    // forward matrix (IDentityt)
  MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, measuredData);
  // check result
  EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - different sizes
//TODO: waiting on text fix from @jcollfont
TEST_F(TikhonovFunctionalTest, DISABLED_loadIDSquareFwdMatrixANDRandDataDiffSizes)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
  MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, measuredData);
  // check result
  EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID non-square fwd matrix + RAND measured data  - different sizes
TEST_F(TikhonovFunctionalTest, loadIDNonSquareFwdMatrixANDRandDataDiffSizes)
{
  // create inputs
  auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
  MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 4)));    // forward matrix (IDentityt)
  MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(5, 1)));   // measurement data (rand)

  // input data
  stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
  stubPortNWithThisData(tikAlgImp, 2, measuredData);
  // check result
  EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

//// ---------- Source Regularization Matrix Input Tests ----------- //////

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDSquaredSourceReguWithNonSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix non-squared option:"non-squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredSourceReguWithNonSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDSquaredSourceReguWithSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix non-squared option:"squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredSourceReguWithSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    Core::Algorithms::AlgorithmParameterName TikhonovSolutionSubcase;
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, 1);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDSquaredSourceReguWithNonSquareOptionWrongSizeWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredSourceReguWithNonSquareOptionWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDSquaredSourceReguWithSquareOptionWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredSourceReguWithSquareOptionWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovSolutionSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmSolutionSubcase::solution_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}


//// ---------- Measurement (residual) Regularization Matrix Input Tests ----------- //////

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDSquaredMeasurementReguWithNonSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix non-squared option:"non-squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredMeasurementReguWithNonSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDSquaredMeasurementReguWithSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix non-squared option:"squared - ok size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredMeasurementReguWithSquareOption)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 4)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDSquaredMeasurementReguWithNonSquareOptionWrongSizeWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredMeasurementReguWithNonSquareOptionWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDSquaredMeasurementReguWithSquareOptionWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(5, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

// ID square fwd matrix + RAND measured data  - ok sizes - Regularization matrix squared option:"non-squared - wrong size
TEST_F(TikhonovFunctionalTest, loadIDNonSquaredMeasurementReguWithSquareOptionWrongSize)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(4, 4)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(4, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Identity(4, 5)));    // forward matrix (Identity)
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(Parameters::TikhonovResidualSubcase, BioPSE::TikhonovAlgorithmImpl::AlgorithmResidualSubcase::residual_constrained_squared);  // select single lambda
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, sourceRegularizationMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
}

/*  TODO: implement functionality tests   */
/// -------- BASIC FUNCTIONS TESTS ------------ ///
/*
// ID square forward matrix with ZERO regularization, RAND input
TEST_F(TikhonovFunctionalTest, functionTestIDFwdMatrixANDRandData)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    DenseMatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
    DenseMatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    DenseMatrixHandle inverseSolution_;
    DenseMatrix measuredDataDesnse = matrix_cast::as_dense(measuredData);
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::RegularizationMethod, std::string("single"));  // select single lambda
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::LambdaFromDirectEntry, 0 );                    // change lambda
    
    // execute
    tikAlgImp->execute();
    
    // return value
    inverseSolution_ = matrix_cast::as_dense(getDataOnThisOutputPort(tikAlgImp,0)_;
    
    // check result
    EXPECT_MATRIX_EQ(inverseSolution_, measuredDataDense);
    
}
*/
/*
// ID square forward matrix with dafault options
TEST_F(TikhonovFunctionalTest, functionTestIDFwdMatrixANDRandDataWithDefaultOptions)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    MatrixHandle inverseSolution;
    DenseMatrix residual;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // change params no change: DEFAULT
    
    // execute
    tikAlgImp->execute();
    
    inverseSolution = getDataOnThisOutputPort(tikAlgImp,0);
    residual = *inverseSolution - measuredData;
 
    // check result
    ASSERT_NEAR( residual.norm(), 0,  abs_error );
}

// Singular forward matrix with regularization
TEST_F(TikhonovFunctionalTest, functionTestZeroFwdMatrixANDRandDataSomeRegu)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Zero(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    MatrixHandle inverseSolution;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::RegularizationMethod, std::string("single"));  // select single lambda
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::LambdaFromDirectEntry, 10 );                    // change lambda
    
    // execute
    tikAlgImp->execute();
    
    inverseSolution = getDataOnThisOutputPort(tikAlgImp,0);
 
    // check result
    ASSERT_NEAR( inverseSolution->norm(), 0,  abs_error );
}


// Singular forward matrix with 0 regularization
TEST_F(TikhonovFunctionalTest, functionTestZeroFwdMatrixANDRandDataNoRegu)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Zero(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::RegularizationMethod, std::string("single"));  // select single lambda
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::LambdaFromDirectEntry, 0 );                    // change lambda
    
    // execute
    tikAlgImp->execute();
    
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::LinearAlgebra::LapackError);
}


// ID forward matrix with source regularization matrix and single lambda
TEST_F(TikhonovFunctionalTest, functionTestIDFwdMatrixANDRandDataInputIDSourceReguSingleLambda)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Zero(3, 3)));    // source regularization matrix (identity)
    MatrixHandle inverseSolution;
    DenseMatrix residual;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::RegularizationMethod, std::string("single"));  // select single lambda
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::LambdaFromDirectEntry, 1 );                    // change lambda
    
    // execute
    tikAlgImp->execute();
    
    inverseSolution = getDataOnThisOutputPort(tikAlgImp,0);
    residual = *inverseSolution - measuredData;
    
    // check result
    ASSERT_NEAR( residual.norm(), 0,  abs_error );

}

// ID forward matrix with source regularization matrix and default L-curve
TEST_F(TikhonovFunctionalTest, functionTestIDFwdMatrixANDRandDataInputIDSourceReguSingleLambda)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    MatrixHandle sourceRegularizationMatrix(new DenseMatrix(DenseMatrix::Zero(3, 3)));    // source regularization matrix (identity)
    MatrixHandle inverseSolution;
    DenseMatrix residual;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 1, sourceRegularizationMatrix);
    
    // change params -> DEFAULT
    
    // execute
    tikAlgImp->execute();
    
    inverseSolution = getDataOnThisOutputPort(tikAlgImp,0);
    residual = *inverseSolution - measuredData;

    
    // check result
    ASSERT_NEAR( residual.norm(), 0,  abs_error );
    
}


// ID forward matrix with residual regularization matrix and single lambda
TEST_F(TikhonovFunctionalTest, functionTestIDFwdMatrixANDRandDataInputIDResidualReguWithSingleLambda)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    MatrixHandle residualRegularizationMatrix(new DenseMatrix(DenseMatrix::Zero(3, 3)));    // source regularization matrix (identity)
    MatrixHandle inverseSolution;
    DenseMatrix residual;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, residualRegularizationMatrix);
    
    // change params
    tikAlgImp->setStateDefaults();                                                  // set default params
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::RegularizationMethod, std::string("single"));  // select single lambda
    tikAlgImp->get_state()->setValue(SolveInverseProblemWithTikhonov::LambdaFromDirectEntry, 1 );                    // change lambda
    
    // execute
    tikAlgImp->execute();
    
    inverseSolution = getDataOnThisOutputPort(tikAlgImp,0);
    residual = *inverseSolution - measuredData;

    
    // check result
    ASSERT_NEAR( residual.norm(), 0,  abs_error );
    
}

// ID forward matrix with residual regularization matrix and default L-curve
TEST_F(TikhonovFunctionalTest, functionTestIDFwdMatrixANDRandDataInputIDResidualReguWithDefaultOpts)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix(new DenseMatrix(DenseMatrix::Identity(3, 3)));    // forward matrix (IDentityt)
    MatrixHandle measuredData(new DenseMatrix(DenseMatrix::Random(3, 1)));   // measurement data (rand)
    MatrixHandle residualRegularizationMatrix(new DenseMatrix(DenseMatrix::Zero(3, 3)));    // source regularization matrix (identity)
    MatrixHandle inverseSolution;
    DenseMatrix residual;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, residualRegularizationMatrix);
    
    // change params -> DEFAULT
    
    // execute
    tikAlgImp->execute();
    
    inverseSolution = getDataOnThisOutputPort(tikAlgImp,0);
    residual = *inverseSolution - measuredData;

    
    // check result
    ASSERT_NEAR( residual.norm(), 0,  abs_error );
    
}
 */

