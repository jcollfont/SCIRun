/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2015 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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

#ifndef BioPSE_TikhonovAlgoAbstractBase_H__
#define BioPSE_TikhonovAlgoAbstractBase_H__

#include <Core/Algorithms/Base/AlgorithmBase.h>
#include <Core/Algorithms/Field/share.h>


namespace SCIRun {
namespace Core {
namespace Algorithms {
namespace Inverse {

	// ALGORITHM_PARAMETER_DECL(RegularizationMethod);
	// ALGORITHM_PARAMETER_DECL(regularizationChoice);
	// ALGORITHM_PARAMETER_DECL(regularizationSolutionSubcase);
	// ALGORITHM_PARAMETER_DECL(regularizationResidualSubcase);
	// ALGORITHM_PARAMETER_DECL(LambdaFromDirectEntry);
	// ALGORITHM_PARAMETER_DECL(LambdaMin);
	// ALGORITHM_PARAMETER_DECL(LambdaMax);
	// ALGORITHM_PARAMETER_DECL(LambdaNum);
	// ALGORITHM_PARAMETER_DECL(LambdaResolution);
	// ALGORITHM_PARAMETER_DECL(LambdaSliderValue);
	// ALGORITHM_PARAMETER_DECL(LambdaCorner);
	// ALGORITHM_PARAMETER_DECL(LCurveText);

	class SCISHARE TikhonovAlgoAbstractBase : public AlgorithmBase
	{

	public:

		// define input names
		static AlgorithmInputName ForwardMatrix;
		static AlgorithmInputName MeasuredPotentials;
		static AlgorithmInputName WeightingInSourceSpace;
		static AlgorithmInputName WeightingInSensorSpace;

		// define parameter names
		static AlgorithmParameterName RegularizationMethod;
		static AlgorithmParameterName regularizationChoice;
		static AlgorithmParameterName regularizationSolutionSubcase;
		static AlgorithmParameterName regularizationResidualSubcase;
		static AlgorithmParameterName LambdaFromDirectEntry;
		static AlgorithmParameterName LambdaMin;
		static AlgorithmParameterName LambdaMax;
		static AlgorithmParameterName LambdaNum;
		static AlgorithmParameterName LambdaResolution;
		static AlgorithmParameterName LambdaSliderValue;
		static AlgorithmParameterName LambdaCorner;
		static AlgorithmParameterName LCurveText;

		// Define algorithm choices
		enum AlgorithmChoice {
			automatic,
			underdetermined,
			overdetermined
		};
		enum AlgorithmSolutionSubcase {
			solution_constrained,
			solution_constrained_squared
		};
		enum AlgorithmResidualSubcase {
			residual_constrained,
			residual_constrained_squared
		};

		struct SCISHARE LCurveInput
		{
			const std::vector<double> rho_;
			const std::vector<double> eta_;
			const std::vector<double> lambdaArray_;
			const int nLambda_;

			LCurveInput(const std::vector<double>& rho, const std::vector<double>& eta, const std::vector<double>& lambdaArray, 			const int nLambda);
		};


		// constructor
		TikhonovAlgoAbstractBase();

		// run function
		virtual AlgorithmOutput run(const AlgorithmInput &) const override;

		// defined public functions
		void update_graph( const AlgorithmInput & input,  double lambda, int lambda_index, const double epsilon);
		static double FindCorner( const AlgorithmInput & input, int& lambda_index);
		static double LambdaLookup(const AlgorithmInput & input, double lambda, int& lambda_index, const double epsilon);
		double computeLcurve(  const AlgorithmInput & input );

	protected:
		// defined protected functions
		double computeLcurve( LCurveInput & input, SCIRun::Core::Datatypes::DenseMatrix& M1, SCIRun::Core::Datatypes::DenseMatrix& M2, SCIRun::Core::Datatypes::DenseMatrix& M3, SCIRun::Core::Datatypes::DenseMatrix& M4, SCIRun::Core::Datatypes::DenseColumnMatrix& y );

		// Abstract functions
		virtual SCIRun::Core::Datatypes::DenseColumnMatrix computeInverseSolution( double lambda_sq, bool inverseCalculation) = 0;

		virtual bool checkInputMatrixSizes( const AlgorithmInput & input );
		virtual void preAlocateInverseMatrices(SCIRun::Core::Datatypes::DenseMatrix& forwardMatrix_,SCIRun::Core::Datatypes::DenseMatrix& 		measuredData_ ,SCIRun::Core::Datatypes::DenseMatrix& sourceWeighting_,SCIRun::Core::Datatypes::DenseMatrix& sensorWeighting_) = 0;
	};

	}}}}

#endif