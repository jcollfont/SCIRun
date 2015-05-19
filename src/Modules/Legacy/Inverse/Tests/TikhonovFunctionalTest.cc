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
using ::testing::_;
using ::testing::NiceMock;
using ::testing::DefaultValue;
using ::testing::Return;


class TikhonovFunctionalTest : public ModuleTest
{
};


// NULL fwd matrix + NULL measure data
TEST_F(TikhonovFunctionalTest,loadNullFwdMatrixANDNullData)
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
TEST_F(TikhonovFunctionalTest,loadIDFwdMatrixANDNullData)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle nullColumnMatrix;              // measurement data (null)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, nullColumnMatrix);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), NullHandleOnPortException);
    
}

// NULL fwd matrix + RANF measured data
TEST_F(TikhonovFunctionalTest,loadNullFwdMatrixANDRandData)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix;    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );    // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), NullHandleOnPortException);
    
}

// ID fwd matrix + RAND measured data
TEST_F(TikhonovFunctionalTest,loadIDFwdMatrixANDRandData)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    
}

// ID non-square fwd matrix + RAND measured data  (underdetermined)
// TODO: FAILS TEST: fails test when it shouldn't. The sizes of forward matrix and data are the same
TEST_F(TikhonovFunctionalTest,loadIDNonSquareFwdMatrixANDRandData)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,4)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(4,1)) );   // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    
}

// ID non-square fwd matrix + RAND measured data  (overdetermined)
// TODO: FAILS TEST: fails test when it shouldn't. The sizes of forward matrix and data are the same
TEST_F(TikhonovFunctionalTest,loadIDNonSquareFwdMatrixANDRandData2)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(4,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    
}

// ID square fwd matrix + RAND measured data  - different sizes
// TODO: FAILS TEST: does not fail test when it shouldn't. The sizes of forward matrix and data are the different (note that this is only for size(fwd,2) < size(data,1) )!
TEST_F(TikhonovFunctionalTest,loadIDSquareFwdMatrixANDRandDataDiffSizes)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(4,1)) );   // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_THROW(tikAlgImp->execute(), SCIRun::Core::DimensionMismatch);
    
}

// ID non-square fwd matrix + RAND measured data  - different sizes
TEST_F(TikhonovFunctionalTest,loadIDNonSquareFwdMatrixANDRandDataDiffSizes)
{
    // create inputs
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,4)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_THROW(tikAlgImp->execute(),SCIRun::Core::DimensionMismatch);
    
}


// -------- REGULARIZATION INPUTS TESTS -------------

// NULL TO SOURCE REGULARIZATION (should be ok)
// TODO: FAILS TEST: should it throw when a NULL input is given or should it go as if there was no input?
TEST_F(TikhonovFunctionalTest,sourceRegularizationNULL)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle sourceReguMatrix;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    
}

// NULL TO RESIDUAL REGULARIZATION (should be ok)
// TODO: FAILS TEST: should it throw when a NULL input is given or should it go as if there was no input?
TEST_F(TikhonovFunctionalTest,residualRegularizationNULL)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix;
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, residualReguMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    
}

// NULL TO SOURCE AND RESIDUAL REGULARIZATION (should be ok)
// TODO: FAILS TEST: should it throw when a NULL input is given or should it go as if there was no input?
TEST_F(TikhonovFunctionalTest,sourceRegularizationNULLresidualRegularizationNULL)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix, sourceReguMatrix;                        // NULL source and residual regularization
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    stubPortNWithThisData(tikAlgImp, 3, residualReguMatrix);
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    
}

// IDENTITY TO SOURCE SIZE_square_ok (no input should give same solution than ID)
TEST_F(TikhonovFunctionalTest,sourceRegularizationID)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle sourceReguMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
    ASSERT_EQ(getDataOnThisOutputPort(tikAlgImp, 0), getDataOnThisOutputPort(tikAlgImpIDregu, 0));
    
}

// IDENTITY TO SOURCE SIZE_square_ok (no input should give different solution than RAND)
// TODO: Tikhonov without regularization matrix in the input (inside implements ID) should be different than any random matrix. If this test fails, means that sourceRegularizationID is not testing what it should be testing or there is something wrong in the algorithm
TEST_F(TikhonovFunctionalTest,sourceRegularizationRAND)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle sourceReguMatrix( new DenseMatrix(DenseMatrix::Random(3,3)) );
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
    ASSERT_NE(getDataOnThisOutputPort(tikAlgImp, 0), getDataOnThisOutputPort(tikAlgImpIDregu, 0));
    
}

// IDENTITY TO RESIDUAL SIZE_square_ok (should give same solution that ID)
TEST_F(TikhonovFunctionalTest,residualRegularizationID)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    stubPortNWithThisData(tikAlgImpIDregu, 3, residualReguMatrix);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
    ASSERT_EQ(getDataOnThisOutputPort(tikAlgImp, 0), getDataOnThisOutputPort(tikAlgImpIDregu, 0));
    
}

// IDENTITY TO RESIDUAL SIZE_square_ok (should give same solution that ID)
// TODO: Tikhonov without regularization matrix in the input (inside implements ID) should be different than any random matrix. If this test fails, means that residualRegularizationID is not testing what it should be testing or there is something wrong in the algorithm
TEST_F(TikhonovFunctionalTest,residualRegularizationRAND)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix( new DenseMatrix(DenseMatrix::Random(3,3)) );
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    stubPortNWithThisData(tikAlgImpIDregu, 3, residualReguMatrix);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
    ASSERT_NE(getDataOnThisOutputPort(tikAlgImp, 0), getDataOnThisOutputPort(tikAlgImpIDregu, 0));
    
}

// IDENTITY TO SOURCE size_rect_ok (should be ok)
TEST_F(TikhonovFunctionalTest,sourceRegularizationID_recSizeOK)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle sourceReguMatrix( new DenseMatrix(DenseMatrix::Identity(4,3)) );
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
    ASSERT_EQ(getDataOnThisOutputPort(tikAlgImp, 0), getDataOnThisOutputPort(tikAlgImpIDregu, 0));
    
}

// IDENTITY TO RESIDUAL size_rect_ok (should be ok)
// TODO: could be a problem with the type of regularization parameter. How can I change it?
TEST_F(TikhonovFunctionalTest,residualRegularizationID_recSizeOK)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix( new DenseMatrix(DenseMatrix::Identity(4,3)) );
    
    // input data
    stubPortNWithThisData(tikAlgImp, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImp, 2, measuredData);
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    stubPortNWithThisData(tikAlgImpIDregu, 3, residualReguMatrix);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImp->execute());
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
    ASSERT_EQ(getDataOnThisOutputPort(tikAlgImp, 0), getDataOnThisOutputPort(tikAlgImpIDregu, 0));
    
}

// IDENTITY TO SOURCE size_square_ko (should complain)
// TODO: could be a problem with the type of regularization parameter. How can I change it?
TEST_F(TikhonovFunctionalTest,sourceRegularizationID_squareSizeKO)
{
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle sourceReguMatrix( new DenseMatrix(DenseMatrix::Identity(4,4)) );
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    
    // check result
    EXPECT_THROW(tikAlgImpIDregu->execute(),SCIRun::Core::DimensionMismatch);
        
}

// IDENTITY TO RESIDUAL size_square_ko (should complain)
TEST_F(TikhonovFunctionalTest,residualRegularizationID_squareSizeKO)
{
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix( new DenseMatrix(DenseMatrix::Identity(4,4)) );
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    stubPortNWithThisData(tikAlgImpIDregu, 3, residualReguMatrix);
    
    
    // check result
    EXPECT_THROW(tikAlgImpIDregu->execute(),SCIRun::Core::DimensionMismatch);
    
}
/*
// RAND+ID TO SOURCE size_square_ok (should be ok)
// TODO: don't know how to add matrices of the type MatrixHandle
TEST_F(TikhonovFunctionalTest,sourceRegularizationIDandRAND_squareSizeOK)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle idReguMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );
    MatrixHandle randReguMatrix( new DenseMatrix(DenseMatrix::Random(3,3)) );
    
    auto sourceReguMatrix = idReguMatrix.data() + *randReguMatrix;
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 1, sourceReguMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
}

// RAND+ID TO RESIDUAL size_square_ok (should be ok)
TEST_F(TikhonovFunctionalTest,residualRegularizationIDandRAND_squareSizeOK)
{
    auto tikAlgImp = makeModule("SolveInverseProblemWithTikhonov");
    auto tikAlgImpIDregu = makeModule("SolveInverseProblemWithTikhonov");
    MatrixHandle fwdMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );    // forward matrix (IDentityt)
    MatrixHandle measuredData( new DenseMatrix(DenseMatrix::Random(3,1)) );   // measurement data (rand)
    MatrixHandle residualReguMatrix( new DenseMatrix(DenseMatrix::Identity(3,3)) );
    
    // input data with ID regularization
    stubPortNWithThisData(tikAlgImpIDregu, 0, fwdMatrix);
    stubPortNWithThisData(tikAlgImpIDregu, 2, measuredData);
    stubPortNWithThisData(tikAlgImpIDregu, 3, residualReguMatrix);
    
    
    // check result
    EXPECT_NO_THROW(tikAlgImpIDregu->execute());
    
}
*/
// SOURCE REGU WITH NULL SPACE ( ? )
// RESIDUAL REGU WITH NULL SPACE ( ? )