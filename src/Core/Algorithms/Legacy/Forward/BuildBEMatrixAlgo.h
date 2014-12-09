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


/*
*  BuildBEMatrix.h:  class to build Boundary Elements matrix
*
*  Written by:
*   Saeed Babaeizadeh
*   Northeastern University
*   January 2006
*/

#ifndef CORE_ALGORITHMS_LEGACY_FORWARD_BUILDBEMATRIXALGO_H
#define CORE_ALGORITHMS_LEGACY_FORWARD_BUILDBEMATRIXALGO_H

#include <Core/Datatypes/MatrixFwd.h>
#include <Core/GeometryPrimitives/GeomFwd.h>
#include <Core/Datatypes/Legacy/Field/FieldFwd.h>
#include <Core/Algorithms/Legacy/Forward/share.h>

namespace SCIRun {
  namespace Core {
    namespace Algorithms {
      namespace Forward {

        class SCISHARE BuildBEMatrixBase
        {
        private:
          static void get_g_coef( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            double,
            double,
            const Geometry::Vector&,
            Datatypes::DenseMatrix&);

          static void get_cruse_weights( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            double,
            double,
            double,
            Datatypes::DenseMatrix& );

          static void getOmega( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            Datatypes::DenseMatrix& );

          static double do_radon_g( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            double,
            double,
            Datatypes::DenseMatrix& );

          static void get_auto_g( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            unsigned int,
            Datatypes::DenseMatrix&,
            double,
            double,
            Datatypes::DenseMatrix& );

          static void bem_sing( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector&,
            unsigned int,
            Datatypes::DenseMatrix&,
            double,
            double,
            Datatypes::DenseMatrix& );

          static double get_new_auto_g( const Geometry::Vector&,
            const Geometry::Vector&,
            const Geometry::Vector& );

        public:
          static void make_cross_G( VMesh*,
            VMesh*,
            Datatypes::DenseMatrixHandle&,
            double,
            double,
            double,
            const std::vector<double>& );

          static void make_auto_G( VMesh*,
            Datatypes::DenseMatrixHandle&,
            double,
            double,
            double,
            const std::vector<double>& );

          static void make_auto_P( VMesh*,
            Datatypes::DenseMatrixHandle&,
            double,
            double,
            double );

          static void make_cross_P( VMesh*,
            VMesh*,
            Datatypes::DenseMatrixHandle&,
            double,
            double,
            double );

          static void pre_calc_tri_areas(VMesh*, std::vector<double>&);
        };


      }}}}

#endif