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

#include <Interface/Modules/Fields/EditMeshBoundingBoxDialog.h>
#include <Modules/Fields/EditMeshBoundingBox.h>

using namespace SCIRun::Gui;
using namespace SCIRun::Dataflow::Networks;

typedef SCIRun::Modules::Fields::EditMeshBoundingBox EditMeshBoundingBoxModule;

EditMeshBoundingBoxDialog::EditMeshBoundingBoxDialog(const std::string& name, ModuleStateHandle state,
  QWidget* parent /* = 0 */)
  : ModuleDialogGeneric(state, parent), doubleUpScale_(1.5625), upScale_(1.25), downScale_(0.8), doubleDownScale_(0.64)
{
  setupUi(this);
  //custom value for cylinder size
  setWindowTitle(QString::fromStdString(name));
  fixSize();

  addCheckBoxManager(useOutputCenterCheckBox_, EditMeshBoundingBoxModule::UseOutputCenter);
  addCheckBoxManager(useOutputSizeCheckBox_, EditMeshBoundingBoxModule::UseOutputSize);
  addCheckableButtonManager(noTranslationRadioButton_, EditMeshBoundingBoxModule::NoTranslation);
  addCheckableButtonManager(xyzTranslationRadioButton_, EditMeshBoundingBoxModule::XYZTranslation);
  addCheckableButtonManager(rdiTranslationRadioButton_, EditMeshBoundingBoxModule::RDITranslation);
  addDoubleSpinBoxManager(outputCenterXSpinBox_, EditMeshBoundingBoxModule::OutputCenterX);
  addDoubleSpinBoxManager(outputCenterYSpinBox_, EditMeshBoundingBoxModule::OutputCenterY);
  addDoubleSpinBoxManager(outputCenterZSpinBox_, EditMeshBoundingBoxModule::OutputCenterZ);
  addDoubleSpinBoxManager(outputSizeXSpinBox_, EditMeshBoundingBoxModule::OutputSizeX);
  addDoubleSpinBoxManager(outputSizeYSpinBox_, EditMeshBoundingBoxModule::OutputSizeY);
  addDoubleSpinBoxManager(outputSizeZSpinBox_, EditMeshBoundingBoxModule::OutputSizeZ);
  addDynamicLabelManager(inputCenterXLabel_, EditMeshBoundingBoxModule::InputCenterX);
  addDynamicLabelManager(inputCenterYLabel_, EditMeshBoundingBoxModule::InputCenterY);
  addDynamicLabelManager(inputCenterZLabel_, EditMeshBoundingBoxModule::InputCenterZ);
  addDynamicLabelManager(inputSizeXLabel_, EditMeshBoundingBoxModule::InputSizeX);
  addDynamicLabelManager(inputSizeYLabel_, EditMeshBoundingBoxModule::InputSizeY);
  addDynamicLabelManager(inputSizeZLabel_, EditMeshBoundingBoxModule::InputSizeZ);

  connect(upScaleToolButton_, SIGNAL(clicked()), this, SLOT(ScaleUpPush()));
  connect(doubleUpScaleToolButton_, SIGNAL(clicked()), this, SLOT(ScaleDoubleUpPush()));
  connect(downScaleToolButton_, SIGNAL(clicked()), this, SLOT(ScaleDownPush()));
  connect(doubleDownScaleToolButton_, SIGNAL(clicked()), this, SLOT(ScaleDoubleDownPush()));


  createExecuteInteractivelyToggleAction();
}

void EditMeshBoundingBoxDialog::AdjustScale(float scaleFactor)
{
  auto scale = state_->getValue(EditMeshBoundingBoxModule::Scale).toDouble();
  scale *= scaleFactor;
  state_->setValue(EditMeshBoundingBoxModule::Scale, scale);
  executeActionTriggered();
}

void EditMeshBoundingBoxDialog::ScaleUpPush(){ AdjustScale(upScale_); }
void EditMeshBoundingBoxDialog::ScaleDoubleUpPush(){ AdjustScale(doubleUpScale_); }
void EditMeshBoundingBoxDialog::ScaleDownPush(){ AdjustScale(downScale_); }
void EditMeshBoundingBoxDialog::ScaleDoubleDownPush(){ AdjustScale(doubleDownScale_); }