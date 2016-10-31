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

#include <sstream>
#include <QtGui>
#include <Interface/Application/NetworkEditor.h>
#include <Interface/Application/Node.h>
#include <Interface/Application/Connection.h>
#include <Interface/Application/ModuleWidget.h>
#include <Interface/Application/ModuleProxyWidget.h>
#include <Interface/Application/Utility.h>
#include <Interface/Application/Port.h>
#include <Interface/Application/PortWidgetManager.h>
#include <Interface/Application/GuiLogger.h>
#include <Interface/Application/NetworkEditorControllerGuiProxy.h>
#include <Interface/Application/ClosestPortFinder.h>
#include <Dataflow/Serialization/Network/NetworkDescriptionSerialization.h>
#include <Dataflow/Engine/Controller/NetworkEditorController.h> //TODO: remove
#include <Dataflow/Network/Module.h> //TODO: remove
#include <Core/Application/Preferences/Preferences.h>
#include <Core/Application/Application.h>
#include <Dataflow/Serialization/Network/XMLSerializer.h>
#ifdef BUILD_WITH_PYTHON
#include <Dataflow/Engine/Python/NetworkEditorPythonAPI.h>
#endif

#include <boost/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/algorithm/string/find.hpp>

using namespace SCIRun;
using namespace SCIRun::Core;
using namespace SCIRun::Gui;
using namespace SCIRun::Dataflow::Networks;
using namespace SCIRun::Dataflow::Engine;

NetworkEditor::NetworkEditor(boost::shared_ptr<CurrentModuleSelection> moduleSelectionGetter,
  boost::shared_ptr<DefaultNotePositionGetter> dnpg,
  boost::shared_ptr<DialogErrorControl> dialogErrorControl,
  PreexecuteFunc preexecuteFunc,
  TagColorFunc tagColor,
  TagNameFunc tagName,
  double highResolutionExpandFactor,
  QWidget* parent)
  : QGraphicsView(parent),
  modulesSelectedByCL_(false),
  currentScale_(1),
  tagLayerActive_(false),
  tagColor_(tagColor),
  tagName_(tagName),
  scene_(new QGraphicsScene(parent)),
  visibleItems_(true),
  lastModulePosition_(0,0),
  dialogErrorControl_(dialogErrorControl),
  moduleSelectionGetter_(moduleSelectionGetter),
  defaultNotePositionGetter_(dnpg),
  moduleEventProxy_(new ModuleEventProxy),
  zLevelManager_(new ZLevelManager(scene_)),
  fileLoading_(false),
  preexecute_(preexecuteFunc),
  highResolutionExpandFactor_(highResolutionExpandFactor)
{
  scene_->setBackgroundBrush(Qt::darkGray);
  ModuleWidget::connectionFactory_.reset(new ConnectionFactory(scene_));
  ModuleWidget::closestPortFinder_.reset(new ClosestPortFinder(scene_));
  ModuleWidget::highResolutionExpandFactor_ = highResolutionExpandFactor_;

  setScene(scene_);
  setDragMode(RubberBandDrag);
  setRenderHints(QPainter::Antialiasing | QPainter::TextAntialiasing);

  connect(scene_, SIGNAL(changed(const QList<QRectF>&)), this, SIGNAL(sceneChanged(const QList<QRectF>&)));

  setSceneRect(QRectF(-1000, -1000, 2000, 2000));
  setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
  setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
  centerOn(100, 100);

  setMouseAsDragMode();

#ifdef BUILD_WITH_PYTHON
  NetworkEditorPythonAPI::setExecutionContext(this);
#endif

  connect(this, SIGNAL(moduleMoved(const SCIRun::Dataflow::Networks::ModuleId&, double, double)), this, SLOT(redrawTagGroups()));
}

void NetworkEditor::setNetworkEditorController(boost::shared_ptr<NetworkEditorControllerGuiProxy> controller)
{
  if (controller_ == controller)
    return;

  if (controller_)
  {
    disconnect(controller_.get(), SIGNAL(moduleAdded(const std::string&, SCIRun::Dataflow::Networks::ModuleHandle, const SCIRun::Dataflow::Engine::ModuleCounter&)),
      this, SLOT(addModuleWidget(const std::string&, SCIRun::Dataflow::Networks::ModuleHandle, const SCIRun::Dataflow::Engine::ModuleCounter&)));

    disconnect(this, SIGNAL(connectionDeleted(const SCIRun::Dataflow::Networks::ConnectionId&)),
      controller_.get(), SLOT(removeConnection(const SCIRun::Dataflow::Networks::ConnectionId&)));

    disconnect(controller_.get(), SIGNAL(connectionAdded(const SCIRun::Dataflow::Networks::ConnectionDescription&)),
      this, SLOT(connectionAddedQueued(const SCIRun::Dataflow::Networks::ConnectionDescription&)));
  }

  controller_ = controller;

  if (controller_)
  {
    connect(controller_.get(), SIGNAL(moduleAdded(const std::string&, SCIRun::Dataflow::Networks::ModuleHandle, const SCIRun::Dataflow::Engine::ModuleCounter&)),
      this, SLOT(addModuleWidget(const std::string&, SCIRun::Dataflow::Networks::ModuleHandle, const SCIRun::Dataflow::Engine::ModuleCounter&)));

    connect(this, SIGNAL(connectionDeleted(const SCIRun::Dataflow::Networks::ConnectionId&)),
      controller_.get(), SLOT(removeConnection(const SCIRun::Dataflow::Networks::ConnectionId&)));

    connect(controller_.get(), SIGNAL(connectionAdded(const SCIRun::Dataflow::Networks::ConnectionDescription&)),
      this, SLOT(connectionAddedQueued(const SCIRun::Dataflow::Networks::ConnectionDescription&)));
  }
}

boost::shared_ptr<NetworkEditorControllerGuiProxy> NetworkEditor::getNetworkEditorController() const
{
  return controller_;
}

void NetworkEditor::addModuleWidget(const std::string& name, ModuleHandle module, const ModuleCounter& count)
{
  latestModuleId_ = module->get_id().id_;
  auto moduleWidget = new ModuleWidget(this, QString::fromStdString(name), module, dialogErrorControl_);
  moduleEventProxy_->trackModule(module);

  setupModuleWidget(moduleWidget);
  if (!fileLoading_)
  {
    moduleWidget->postLoadAction();
  }
  count.increment();
  Q_EMIT modified();
  Q_EMIT newModule(QString::fromStdString(module->get_id()), module->has_ui());
}

void NetworkEditor::connectionAddedQueued(const ConnectionDescription& cd)
{
  //std::cout << "Received queued connection request: " << ConnectionId::create(cd).id_ << std::endl;
}

boost::shared_ptr<DisableDynamicPortSwitch> NetworkEditor::createDynamicPortDisabler()
{
  return controller_->createDynamicPortSwitch();
}

boost::optional<ConnectionId> NetworkEditor::requestConnection(const PortDescriptionInterface* from, const PortDescriptionInterface* to)
{
  auto id = controller_->requestConnection(from, to);
  Q_EMIT modified();
  return id;
}

namespace
{
  const int TagTextKey = 123;

  ModuleProxyWidget* findById(const QList<QGraphicsItem*>& list, const std::string& id)
  {
    Q_FOREACH(QGraphicsItem* item, list)
    {
      if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
      {
        if (id == w->getModuleWidget()->getModuleId())
          return w;
      }
    }
    return nullptr;
  }

  ModuleProxyWidget* findFirstByName(const QList<QGraphicsItem*>& list, const std::string& name)
  {
    Q_FOREACH(QGraphicsItem* item, list)
    {
      if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
      {
        if (w->getModuleWidget()->getModuleId().find(name) != std::string::npos)
          return w;
      }
    }
    return nullptr;
  }
}

void NetworkEditor::duplicateModule(const ModuleHandle& module)
{
  auto widget = findById(scene_->items(), module->get_id());
  lastModulePosition_ = widget->scenePos() + QPointF(0, 110);
  //TODO: need better duplicate placement. hard code it for now.
  controller_->duplicateModule(module);
}

namespace
{
  QPointF moduleAddIncrement(20, 20);
}

void NetworkEditor::connectNewModule(const ModuleHandle& moduleToConnectTo, const PortDescriptionInterface* portToConnect, const std::string& newModuleName)
{
  auto prop = sender()->property(addNewModuleActionTypePropertyName());

  auto widget = findById(scene_->items(), moduleToConnectTo->get_id());
  QPointF increment(0, portToConnect->isInput() ? -110 : 50);
  lastModulePosition_ = widget->scenePos() + increment;
  moduleAddIncrement = { 20.0, portToConnect->isInput() ? -20.0 : 20.0 };

  PortWidget* newConnectionInputPort = nullptr;
  auto q = dynamic_cast<const PortWidget*>(portToConnect);
  if (q)
  {
    for (size_t i = 0; i < q->nconnections(); ++i)
    {
      auto cpi = q->connectedPorts()[i];
      if (QString::fromStdString(cpi->id().toString()) == sender()->property(insertNewModuleActionTypePropertyName()))
        newConnectionInputPort = cpi;
    }
  }

  if (newConnectionInputPort)
  {
    controller_->removeConnection(*newConnectionInputPort->firstConnectionId());
    newConnectionInputPort->deleteConnectionsLater();
  }

  controller_->connectNewModule(portToConnect, newModuleName, newConnectionInputPort);
}

void NetworkEditor::replaceModuleWith(const ModuleHandle& moduleToReplace, const std::string& newModuleName)
{
  auto oldModule = findById(scene_->items(), moduleToReplace->get_id());
  lastModulePosition_ = oldModule->scenePos();
  controller_->addModule(newModuleName);

  // connect up same ports
  auto newModule = findById(scene_->items(), latestModuleId_);

  const auto& oldModPorts = oldModule->getModuleWidget()->ports();
  const auto& newModPorts = newModule->getModuleWidget()->ports();

  {
    int nextInputIndex = 0;
    for (const auto& iport : oldModPorts.inputs())
    {
      if (iport->isConnected())
      {
        const auto& newInputs = newModPorts.inputs();
        auto toConnect = std::find_if(newInputs.begin(), newInputs.end(),
          [&](const PortWidget* port) { return port->get_typename() == iport->get_typename() && port->getIndex() >= nextInputIndex; });
        if (toConnect == newInputs.end())
        {
          qDebug() << "Logical error: could not find input port to connect to" << iport << nextInputIndex;
          break;
        }
        requestConnection(iport->connectedPorts()[0], *toConnect);
        nextInputIndex = (*toConnect)->getIndex() + 1;
      }
    }
  }

  {
    int nextOutputIndex = 0;
    auto newOutputs = newModPorts.outputs();
    for (const auto& oport : oldModPorts.outputs())
    {
      if (oport->isConnected())
      {
        auto toConnect = std::find_if(newOutputs.begin(), newOutputs.end(),
          [&](const PortWidget* port) { return port->get_typename() == oport->get_typename() && port->getIndex() >= nextOutputIndex; });
        if (toConnect == newOutputs.end())
        {
          qDebug() << "Logical error: could not find output port to connect to" << oport;
          break;
        }
        auto connectedPorts = oport->connectedPorts();
        std::vector<PortWidget*> dynamicPortsNeedSpecialHandling;
        std::copy_if(connectedPorts.begin(), connectedPorts.end(), std::back_inserter(dynamicPortsNeedSpecialHandling), [](const PortWidget* p) { return p->isDynamic(); });
        connectedPorts.erase(std::remove_if(connectedPorts.begin(), connectedPorts.end(), [](const PortWidget* p) { return p->isDynamic(); }), connectedPorts.end());
        oport->deleteConnections();
        for (const auto& connected : connectedPorts)
        {
          requestConnection(connected, *toConnect);
        }
        nextOutputIndex = (*toConnect)->getIndex() + 1;
      }
    }
  }

  oldModule->deleteLater();
}

ModuleProxyWidget* NetworkEditor::setupModuleWidget(ModuleWidget* module)
{
  auto proxy = new ModuleProxyWidget(module);

  connect(module, SIGNAL(removeModule(const SCIRun::Dataflow::Networks::ModuleId&)), controller_.get(), SLOT(removeModule(const SCIRun::Dataflow::Networks::ModuleId&)));
  connect(module, SIGNAL(interrupt(const SCIRun::Dataflow::Networks::ModuleId&)), controller_.get(), SLOT(interrupt(const SCIRun::Dataflow::Networks::ModuleId&)));
  connect(module, SIGNAL(removeModule(const SCIRun::Dataflow::Networks::ModuleId&)), this, SIGNAL(modified()));
  connect(module, SIGNAL(noteChanged()), this, SIGNAL(modified()));
  connect(module, SIGNAL(executionDisabled(bool)), this, SIGNAL(modified()));
  connect(module, SIGNAL(requestConnection(const SCIRun::Dataflow::Networks::PortDescriptionInterface*, const SCIRun::Dataflow::Networks::PortDescriptionInterface*)),
    this, SLOT(requestConnection(const SCIRun::Dataflow::Networks::PortDescriptionInterface*, const SCIRun::Dataflow::Networks::PortDescriptionInterface*)));
  connect(module, SIGNAL(duplicateModule(const SCIRun::Dataflow::Networks::ModuleHandle&)), this, SLOT(duplicateModule(const SCIRun::Dataflow::Networks::ModuleHandle&)));
  connect(this, SIGNAL(networkEditorMouseButtonPressed()), module, SIGNAL(cancelConnectionsInProgress()));
  connect(controller_.get(), SIGNAL(connectionAdded(const SCIRun::Dataflow::Networks::ConnectionDescription&)),
    module, SIGNAL(connectionAdded(const SCIRun::Dataflow::Networks::ConnectionDescription&)));
  connect(module, SIGNAL(executedManually(const SCIRun::Dataflow::Networks::ModuleHandle&, bool)),
    this, SLOT(executeModule(const SCIRun::Dataflow::Networks::ModuleHandle&, bool)));
  connect(module, SIGNAL(connectionDeleted(const SCIRun::Dataflow::Networks::ConnectionId&)),
    this, SIGNAL(connectionDeleted(const SCIRun::Dataflow::Networks::ConnectionId&)));
  connect(module, SIGNAL(connectionDeleted(const SCIRun::Dataflow::Networks::ConnectionId&)), this, SIGNAL(modified()));
  connect(module, SIGNAL(connectNewModule(const SCIRun::Dataflow::Networks::ModuleHandle&, const SCIRun::Dataflow::Networks::PortDescriptionInterface*, const std::string&)),
    this, SLOT(connectNewModule(const SCIRun::Dataflow::Networks::ModuleHandle&, const SCIRun::Dataflow::Networks::PortDescriptionInterface*, const std::string&)));
  connect(module, SIGNAL(replaceModuleWith(const SCIRun::Dataflow::Networks::ModuleHandle&, const std::string&)),
    this, SLOT(replaceModuleWith(const SCIRun::Dataflow::Networks::ModuleHandle&, const std::string&)));
  connect(module, SIGNAL(disableWidgetDisabling()), this, SIGNAL(disableWidgetDisabling()));
  connect(module, SIGNAL(reenableWidgetDisabling()), this, SIGNAL(reenableWidgetDisabling()));

  if (module->hasDynamicPorts())
  {
    connect(controller_.get(), SIGNAL(portAdded(const SCIRun::Dataflow::Networks::ModuleId&, const SCIRun::Dataflow::Networks::PortId&)), module, SLOT(addDynamicPort(const SCIRun::Dataflow::Networks::ModuleId&, const SCIRun::Dataflow::Networks::PortId&)));
    connect(controller_.get(), SIGNAL(portRemoved(const SCIRun::Dataflow::Networks::ModuleId&, const SCIRun::Dataflow::Networks::PortId&)), module, SLOT(removeDynamicPort(const SCIRun::Dataflow::Networks::ModuleId&, const SCIRun::Dataflow::Networks::PortId&)));
    connect(module, SIGNAL(dynamicPortChanged(const std::string&, bool)), proxy, SLOT(createPortPositionProviders()));
  }

  LOG_DEBUG("NetworkEditor connecting to state" << std::endl);
  module->getModule()->get_state()->connectStateChanged(boost::bind(&NetworkEditor::modified, this));

  connect(this, SIGNAL(networkExecuted()), module, SLOT(resetLogButtonColor()));
  connect(this, SIGNAL(networkExecuted()), module, SLOT(resetProgressBar()));

  proxy->setZValue(zLevelManager_->get_max());

  while (!scene_->items(lastModulePosition_.x() - 20, lastModulePosition_.y() - 20, 40, 40).isEmpty())
  {
    lastModulePosition_ += moduleAddIncrement;
  }
  proxy->setPos(lastModulePosition_);

  proxy->setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemIsSelectable | QGraphicsItem::ItemSendsGeometryChanges);
  connect(scene_, SIGNAL(selectionChanged()), proxy, SLOT(highlightIfSelected()));
  connect(proxy, SIGNAL(selected()), this, SLOT(bringToFront()));
  connect(proxy, SIGNAL(widgetMoved(const SCIRun::Dataflow::Networks::ModuleId&, double, double)), this, SIGNAL(modified()));
  connect(proxy, SIGNAL(widgetMoved(const SCIRun::Dataflow::Networks::ModuleId&, double, double)), this, SIGNAL(moduleMoved(const SCIRun::Dataflow::Networks::ModuleId&, double, double)));
  connect(this, SIGNAL(snapToModules()), proxy, SLOT(snapToGrid()));
  connect(this, SIGNAL(highlightPorts(int)), proxy, SLOT(highlightPorts(int)));
  connect(this, SIGNAL(resetModulesDueToCycle()), module, SLOT(changeExecuteButtonToPlay()));
  connect(this, SIGNAL(defaultNotePositionChanged(NotePosition)), proxy, SLOT(setDefaultNotePosition(NotePosition)));
  connect(module, SIGNAL(displayChanged()), this, SLOT(updateViewport()));
  connect(module, SIGNAL(displayChanged()), proxy, SLOT(createPortPositionProviders()));
  connect(proxy, SIGNAL(tagChanged(int)), this, SLOT(highlightTaggedItem(int)));

  proxy->setDefaultNotePosition(defaultNotePositionGetter_->position());
  proxy->createPortPositionProviders();
  proxy->highlightPorts(Preferences::Instance().highlightPorts ? 1 : 0);

  if (highResolutionExpandFactor_ > 1)
  {
    //qDebug() << "module widget expand factor:" << highResolutionExpandFactor_;
    //qDebug() << proxy->size();
    module->setFixedHeight(proxy->size().height() * highResolutionExpandFactor_);
    proxy->setMaximumHeight(proxy->size().height() * highResolutionExpandFactor_);
    module->setFixedWidth(proxy->size().width() * std::max(highResolutionExpandFactor_*0.9, 1.0));
    proxy->setMaximumWidth(proxy->size().width() * std::max(highResolutionExpandFactor_*0.9, 1.0));
    //qDebug() << proxy->size();
  }

  scene_->addItem(proxy);
  ensureVisible(proxy);
  proxy->createStartupNote();

  scene_->clearSelection();
  proxy->setSelected(true);
  bringToFront();
  proxy->setVisible(visibleItems_);

  GuiLogger::Instance().logInfoStd("Module added: " + module->getModuleId());

  return proxy;
}

void NetworkEditor::setMouseAsDragMode()
{
  setDragMode(ScrollHandDrag);
}

void NetworkEditor::setMouseAsSelectMode()
{
  setDragMode(RubberBandDrag);
}

void NetworkEditor::bringToFront()
{
  zLevelManager_->bringToFront();
}

void ZLevelManager::bringToFront()
{
  ++maxZ_;
  setZValue(maxZ_);
}

void NetworkEditor::sendToBack()
{
  zLevelManager_->sendToBack();
}

void ZLevelManager::sendToBack()
{
  --minZ_;
  setZValue(minZ_);
}

void ZLevelManager::setZValue(int z)
{
  auto node = selectedModuleProxy();
  if (node)
  {
    node->setZValue(z);
  }
}

ModuleProxyWidget* getModuleProxy(QGraphicsItem* item)
{
  return dynamic_cast<ModuleProxyWidget*>(item);
}

ModuleWidget* getModule(QGraphicsItem* item)
{
  auto proxy = getModuleProxy(item);
  if (proxy)
    return static_cast<ModuleWidget*>(proxy->widget());
  return nullptr;
}

void NetworkEditor::setVisibility(bool visible)
{
  visibleItems_ = visible;
  ModuleWidget::connectionFactory_->setVisibility(visible);
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto p = getModuleProxy(item))
      p->setVisible(visibleItems_);
    else if (auto c = dynamic_cast<ConnectionLine*>(item))
      c->setVisible(visibleItems_);
  }
}

//TODO copy/paste
ModuleWidget* NetworkEditor::selectedModule() const
{
  auto items = scene_->selectedItems();
  if (items.count() == 1)
  {
    return getModule(items.first());
  }
  return nullptr;
}

ModuleProxyWidget* ZLevelManager::selectedModuleProxy() const
{
  auto items = scene_->selectedItems();
  if (items.count() == 1)
  {
    return getModuleProxy(items.first());
  }
  return nullptr;
}

ConnectionLine* NetworkEditor::selectedLink() const
{
  auto items = scene_->selectedItems();
  if (items.count() == 1)
    return dynamic_cast<ConnectionLine*>(items.first());
  return nullptr;
}

NetworkEditor::ModulePair NetworkEditor::selectedModulePair() const
{
  auto items = scene_->selectedItems();
  if (items.count() == 2)
  {
    auto first = getModule(items.first());
    auto second = getModule(items.last());
    if (first && second)
		return ModulePair(first, second);
  }
  return ModulePair();
}

void NetworkEditor::del()
{
  auto items = scene_->selectedItems();
  QMutableListIterator<QGraphicsItem*> i(items);
  while (i.hasNext())
  {
    auto link = dynamic_cast<QGraphicsPathItem*>(i.next());
    if (link)
    {
      scene()->removeItem(link);
      delete link;
      i.remove();
    }
  }
  qDeleteAll(items);
  updateViewport();
  Q_EMIT modified();
}

void NetworkEditor::cut()
{
  copy();
  del();
}

void NetworkEditor::copy()
{
  auto selected = scene_->selectedItems();
  auto modSelected = [&selected](ModuleHandle mod)
  {
    for (const auto& item : selected)
    {
      if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
      {
        if (w->getModuleWidget()->getModuleId() == mod->get_id().id_)
          return true;
      }
    }
    return false;
  };
  auto connSelected = [&selected](const ConnectionDescription& conn)
  {
    for (const auto& item : selected)
    {
      if (auto connLine = dynamic_cast<ConnectionLine*>(item))
      {
        if (connLine->id().describe() == conn)
          return true;
      }
    }
    return false;
  };

  auto file = controller_->serializeNetworkFragment(modSelected, connSelected);

  if (file)
  {
    //TODO encapsulate
    std::ostringstream ostr;
    XMLSerializer::save_xml(*file, ostr, "networkFragment");
    auto xml = QString::fromStdString(ostr.str());

    QApplication::clipboard()->setText(xml);
    Q_EMIT newSubnetworkCopied(xml);
  }
  else
  {
    qDebug() << "null net fragment";
  }
}

void NetworkEditor::paste()
{
  pasteImpl(QApplication::clipboard()->text());
}

void NetworkEditor::pasteImpl(const QString& xml)
{
  std::istringstream istr(xml.toStdString());
  try
  {
    appendToNetwork(XMLSerializer::load_xml<NetworkFile>(istr));
  }
  catch (...)
  {
    QMessageBox::critical(this, "Paste error", "Invalid clipboard contents: " + xml);
  }
}

void NetworkEditor::contextMenuEvent(QContextMenuEvent *event)
{
  //TODO: this menu needs to check for certain editing conditions. Disabling for now.
  if (false)
  {
    auto items = scene_->items(mapToScene(event->pos()));
    if (items.isEmpty())
    {
      QMenu menu(this);
      menu.addActions(actions());
      menu.exec(event->globalPos());
    }
  }
}

void NetworkEditor::dropEvent(QDropEvent* event)
{
  auto data = event->mimeData();
  if (data->hasUrls())
  {
    auto urls = data->urls();
    if (!urls.isEmpty())
    {
      auto file = urls[0].path();
      QFileInfo check_file(file);
      if (check_file.exists() && check_file.isFile() && file.endsWith("srn5"))
      {
        Q_EMIT requestLoadNetwork(file);
        return;
      }
    }
  }

  if (moduleSelectionGetter_->isModule())
  {
    addNewModuleAtPosition(mapToScene(event->pos()));
  }
  else if (moduleSelectionGetter_->isClipboardXML())
    pasteImpl(moduleSelectionGetter_->clipboardXML());
}

void NetworkEditor::addNewModuleAtPosition(const QPointF& position)
{
  lastModulePosition_ = position;
  controller_->addModule(moduleSelectionGetter_->text().toStdString());
  Q_EMIT modified();
}

void NetworkEditor::addModuleViaDoubleClickedTreeItem()
{
  if (moduleSelectionGetter_->isModule())
  {
    auto upperLeft = mapToScene(viewport()->geometry()).boundingRect().center();
    addNewModuleAtPosition(upperLeft);
  }
  else if (moduleSelectionGetter_->isClipboardXML())
    pasteImpl(moduleSelectionGetter_->clipboardXML());
}

void NetworkEditor::dragEnterEvent(QDragEnterEvent* event)
{
  //???
  event->acceptProposedAction();
}

void NetworkEditor::dragMoveEvent(QDragMoveEvent* event)
{
}

void NetworkEditor::updateViewport()
{
  viewport()->update();
}

void NetworkEditor::mouseMoveEvent(QMouseEvent *event)
{
	if (event->button() != Qt::LeftButton)
		Q_EMIT networkEditorMouseButtonPressed();

  if (auto cL = getSingleConnectionSelected())
  {
    if (event->buttons() & Qt::LeftButton)
    {
      if (!(event->modifiers() & Qt::ControlModifier))
      {
        auto selectedPair = cL->getConnectedToModuleIds();

        findById(scene_->items(), selectedPair.first)->setSelected(true);
        findById(scene_->items(), selectedPair.second)->setSelected(true);
        modulesSelectedByCL_ = true;
      }
    }
  }
  QGraphicsView::mouseMoveEvent(event);
}

void NetworkEditor::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::MiddleButton)
    Q_EMIT middleMouseClicked();
  QGraphicsView::mousePressEvent(event);
}

void NetworkEditor::mouseReleaseEvent(QMouseEvent *event)
{
  if (modulesSelectedByCL_)
  {
    unselectConnectionGroup();
    Q_EMIT modified();
  }
  modulesSelectedByCL_ = false;
  QGraphicsView::mouseReleaseEvent(event);
}

NetworkSearchWidget::NetworkSearchWidget(NetworkEditor* ned)
{
  setupUi(this);
  //connect(closeButton_, SIGNAL(clicked()), ned, SLOT(hideSearchBox()));
  connect(searchLineEdit_, SIGNAL(textChanged(const QString&)), ned, SLOT(searchTextChanged(const QString&)));
  connect(clearToolButton_, SIGNAL(clicked()), searchLineEdit_, SLOT(clear()));
}

SearchResultItem::SearchResultItem(const QString& text, const QColor& color, std::function<void()> action, QGraphicsItem* parent)
  : FloatingTextItem(text, action, parent)
{
  setDefaultTextColor(color);
  auto backgroundGray = QString("background:rgba(%1, %1, %1, 30%)").arg(200);
  setHtml("<div style='" + backgroundGray + ";font: 15px Lucida, sans-serif'>" + toPlainText() + "</div>");
  items_.insert(this);
}

SearchResultItem::~SearchResultItem()
{
  items_.erase(this);
}

void SearchResultItem::removeAll()
{
  auto copyOfItems(items_);
  for (auto& sri : copyOfItems)
    delete sri;
  items_.clear();
}

std::set<SearchResultItem*> SearchResultItem::items_;

enum SearchTupleParts
{
  ItemType,
  ItemName,
  ItemAction,
  ItemColor
};

class NetworkSearchEngine
{
public:
  NetworkSearchEngine(QGraphicsScene* scene, TagColorFunc tagColor) : scene_(scene), tagColor_(tagColor) {}

  using Result = std::tuple<QString, QString, std::function<void()>, QColor>;
  using ResultList = std::vector<Result>;
  ResultList search(const QString& text) const
  {
    ResultList results;
    Q_FOREACH(auto item, scene_->items())
    {
      ResultList subresults;
      if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
      {
        subresults = searchItem(w, text);
      }
      else if (dynamic_cast<FloatingTextItem*>(item))
      {
        // skip--don't search errors or search results
      }
      else if (auto t = dynamic_cast<QGraphicsTextItem*>(item))
      {
        subresults = searchItem(t, text);
      }
      else if (auto s = dynamic_cast<QGraphicsSimpleTextItem*>(item))
      {
        subresults = searchItem(s, text);
      }
      else
      {
        //qDebug() << "something else";
      }
      results.insert(results.end(), subresults.begin(), subresults.end());
    }
    std::sort(results.begin(), results.end(), [](const Result& r1, const Result& r2) { return std::get<ItemType>(r1) < std::get<ItemType>(r2); });
    return results;
  }
private:
  ResultList searchItem(ModuleProxyWidget* mod, const QString& text) const
  {
    ResultList results;
    auto id = mod->getModuleWidget()->getModuleId();
    if (boost::ifind_first(id, text.toStdString()))
    {
      auto tag = mod->data(TagDataKey).toInt();
      results.emplace_back("Module",
        QString::fromStdString(id),
        [mod]() { mod->showAndColor(Qt::green); },
        tag != NoTag ? tagColor_(tag) : Qt::white);
    }

    auto metadata = mod->getModuleWidget()->metadataToString();
    if (metadata.contains(text, Qt::CaseInsensitive))
    {
      results.emplace_back("Module metadata match in",
        QString::fromStdString(id),
        [mod]() { mod->showAndColor(Qt::yellow); },
        Qt::yellow);
    }

    auto dialog = mod->getModuleWidget()->dialog();
    if (dialog && text.length() > 5)
    {
      auto widgetMatches = dialog->findChildren<QWidget*>(QRegExp(".*" + text + ".*", Qt::CaseInsensitive));
      Q_FOREACH(auto widget, widgetMatches)
      {
        results.emplace_back("Module UI widget match",
          QString::fromStdString(id) + "::" + widget->objectName(),
          [mod]() { mod->showAndColor("#AA3333"); },
          "#AA3333");
      }
    }
    return results;
  }

  ResultList searchItem(QGraphicsTextItem* note, const QString& text) const
  {
    ResultList results;
    auto cursor = note->document()->find(text);
    if (!cursor.isNull())
    {
      results.emplace_back("Note",
        "..." + note->toPlainText().mid(cursor.position() - 10, 20) + "...",
        [note, text]() { ModuleProxyWidget::ensureItemVisible(note); selectNote(note, text); },
        Qt::white);
    }
    return results;
  }

  ResultList searchItem(QGraphicsSimpleTextItem* tag, const QString& text) const
  {
    ResultList results;
    if (tag->text().contains(text, Qt::CaseInsensitive))
    {
      results.emplace_back("Tag",
        tag->text(),
        [tag]() { ModuleProxyWidget::ensureItemVisible(tag); },
        tagColor_(tag->data(TagTextKey).toInt()));
    }
    return results;
  }

  static void selectNote(QGraphicsTextItem* note, const QString& text)
  {
    auto doc = note->document();
    QTextCursor cur(doc->find(text));
    note->setTextCursor(cur);
  }

  QGraphicsScene* scene_;
  TagColorFunc tagColor_;
};

void NetworkEditor::searchTextChanged(const QString& text)
{
  if (text.isEmpty())
  {
    SearchResultItem::removeAll();
    return;
  }
  if (text.length() > 2)
  {
    SearchResultItem::removeAll();

    NetworkSearchEngine engine(scene(), tagColor_);
    auto results = engine.search(text);
    auto textScale = 1.0 / currentScale_;
    if (!results.empty())
    {
      auto title = new SearchResultItem("Search results:", Qt::green, {});
      title->setPos(positionOfFloatingText(title->num(), true, 20, textScale * 22));
      scene()->addItem(title);
      title->scale(textScale, textScale);
    }
    for (const auto& result : results)
    {
      auto searchItem = new SearchResultItem(std::get<ItemType>(result) + ": " + std::get<ItemName>(result),
        std::get<ItemColor>(result), std::get<ItemAction>(result));
      searchItem->setPos(positionOfFloatingText(searchItem->num(), true, 50, textScale * 22));
      scene()->addItem(searchItem);
      searchItem->scale(textScale, textScale);
    }
  }
}

QPointF NetworkEditor::positionOfFloatingText(int num, bool top, int horizontalIndent, int verticalSpacing) const
{
  QPointF tl(horizontalScrollBar()->value(), verticalScrollBar()->value());
  auto br = tl + viewport()->rect().bottomRight();
  auto mat = matrix().inverted();
  auto rect = mat.mapRect(QRectF(tl, br));

  auto corner = top ? rect.topLeft() : rect.bottomLeft();
  return (corner + QPointF(horizontalIndent, (top ? 1 : -1)* (verticalSpacing * num + 100)));
}

void NetworkEditor::displayError(const QString& msg, std::function<void()> showModule)
{
  if (Preferences::Instance().showModuleErrorInlineMessages)
  {
    auto errorItem = new ErrorItem(msg, showModule);
    scene()->addItem(errorItem);

    errorItem->setPos(positionOfFloatingText(errorItem->num(), false, 30, 40));
  }
}

ConnectionLine* NetworkEditor::getSingleConnectionSelected()
{
	ConnectionLine* connectionSelected = nullptr;
	auto item = scene_->selectedItems();
	if(item.count() == 1 && (connectionSelected = qgraphicsitem_cast<ConnectionLine*>(item.first())))
		return connectionSelected;
	return connectionSelected;
}

void NetworkEditor::unselectConnectionGroup()
{
  auto items = scene_->selectedItems();
	if (items.count() == 3)
	{
		int hasConnection = 0;
		int	hasWidgets = 0;

		Q_FOREACH(QGraphicsItem* item, items)
		{
			if (auto cL = qgraphicsitem_cast<ConnectionLine*>(item))
			{
				++hasConnection;
				items.push_front(cL);
			}
			if (qgraphicsitem_cast<ModuleProxyWidget*>(item))
				++hasWidgets;
		}
		if(hasConnection == 1 && hasWidgets == 2)
		{
			if (auto cL = qgraphicsitem_cast<ConnectionLine*>(items.first()))
			{
				auto selectedPair = cL->getConnectedToModuleIds();

				cL->setSelected(false);
				findById(scene_->items(),selectedPair.first)->setSelected(false);
				findById(scene_->items(),selectedPair.second)->setSelected(false);
			}
		}
	}
}

ModulePositionsHandle NetworkEditor::dumpModulePositions(ModuleFilter filter) const
{
  auto positions(boost::make_shared<ModulePositions>());
  fillModulePositionMap(*positions, filter);
  return positions;
}

void NetworkEditor::fillModulePositionMap(ModulePositions& positions, ModuleFilter filter) const
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      if (filter(w->getModuleWidget()->getModule()))
        positions.modulePositions[w->getModuleWidget()->getModuleId()] = std::make_pair(item->scenePos().x(), item->scenePos().y());
    }
  }
}

void NetworkEditor::centerView()
{
  ModulePositions positions;
  fillModulePositionMap(positions, boost::lambda::constant(true));
  centerOn(findCenterOfNetwork(positions));
}

ModuleNotesHandle NetworkEditor::dumpModuleNotes(ModuleFilter filter) const
{
  auto notes(boost::make_shared<ModuleNotes>());
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      auto note = w->currentNote();
      if (filter(w->getModuleWidget()->getModule()) &&
        !note.plainText_.isEmpty())
        notes->notes[w->getModuleWidget()->getModuleId()] = NoteXML(note.html_.toStdString(), note.position_, note.plainText_.toStdString(), note.fontSize_);
    }
  }
  return notes;
}

void NetworkEditor::copyNote(ModuleHandle from, ModuleHandle to) const
{
  Note note;
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      if (w->getModuleWidget()->getModule()->get_id() == from->get_id())
      {
        note = w->currentNote();
        break;
      }
    }
  }

  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      if (w->getModuleWidget()->getModule()->get_id() == to->get_id())
      {
        w->updateNote(note);
        w->getModuleWidget()->setCurrentNote(note, true);
        break;
      }
    }
  }
}

namespace
{
  std::string connectionNoteId(const ModuleIdPair& ms)
  {
    return ms.first.id_ + "--" + ms.second.id_;
  }
}

ConnectionNotesHandle NetworkEditor::dumpConnectionNotes(ConnectionFilter filter) const
{
  auto notes(boost::make_shared<ConnectionNotes>());
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto conn = dynamic_cast<ConnectionLine*>(item))
    {
      auto note = conn->currentNote();
      if (filter(conn->id().describe()) &&
        !note.plainText_.isEmpty())
      {
        //TODO hacky
        auto id = connectionNoteId(conn->getConnectedToModuleIds());
        notes->notes[id] = NoteXML(note.html_.toStdString(), note.position_, note.plainText_.toStdString(), note.fontSize_);
      }
    }
  }
  return notes;
}

ModuleTagsHandle NetworkEditor::dumpModuleTags(ModuleFilter filter) const
{
  auto tags(boost::make_shared<ModuleTags>());
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto mod = dynamic_cast<ModuleProxyWidget*>(item))
    {
      if (filter(mod->getModuleWidget()->getModule()))
        tags->tags[mod->getModuleWidget()->getModuleId()] = mod->data(TagDataKey).toInt();
    }
  }
  tags->labels = tagLabelOverrides_;
  tags->showTagGroupsOnLoad = showTagGroupsOnFileLoad();
  return tags;
}

DisabledComponentsHandle NetworkEditor::dumpDisabledComponents(ModuleFilter modFilter, ConnectionFilter connFilter) const
{
  auto disabled(boost::make_shared<DisabledComponents>());
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto mod = dynamic_cast<ModuleProxyWidget*>(item))
    {
      if (mod->getModuleWidget()->executionDisabled() && modFilter(mod->getModuleWidget()->getModule()))
        disabled->disabledModules.push_back(mod->getModuleWidget()->getModuleId());
    }
    if (auto conn = dynamic_cast<ConnectionLine*>(item))
    {
      if (conn->disabled() && connFilter(conn->id().describe()))
      {
        disabled->disabledConnections.emplace_back(connectionNoteId(conn->getConnectedToModuleIds()));
      }
    }
  }
  return disabled;
}

void NetworkEditor::updateModulePositions(const ModulePositions& modulePositions, bool selectAll)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      auto posIter = modulePositions.modulePositions.find(w->getModuleWidget()->getModuleId());
      if (posIter != modulePositions.modulePositions.end())
      {
        w->setPos(posIter->second.first, posIter->second.second);
        ensureVisible(w);
        if (selectAll)
          w->setSelected(true);
      }
    }
  }
}

void NetworkEditor::updateModuleNotes(const ModuleNotes& moduleNotes)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      auto noteIter = moduleNotes.notes.find(w->getModuleWidget()->getModuleId());
      if (noteIter != moduleNotes.notes.end())
      {
        auto noteXML = noteIter->second;
        Note note(QString::fromStdString(noteXML.noteHTML), QString::fromStdString(noteXML.noteText), noteXML.fontSize, noteXML.position);
        w->getModuleWidget()->updateNoteFromFile(note);
      }
    }
  }
}

void NetworkEditor::updateModuleTags(const ModuleTags& moduleTags)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      auto tagIter = moduleTags.tags.find(w->getModuleWidget()->getModuleId());
      if (tagIter != moduleTags.tags.end())
      {
        w->setData(TagDataKey, tagIter->second);
      }
    }
  }
  setShowTagGroupsOnFileLoad(moduleTags.showTagGroupsOnLoad);
  tagLabelOverrides_ = moduleTags.labels;
  if (showTagGroupsOnFileLoad())
  {
    tagGroupsActive_ = true;
    drawTagGroups();
  }
}

void NetworkEditor::updateConnectionNotes(const ConnectionNotes& notes)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto conn = dynamic_cast<ConnectionLine*>(item))
    {
      auto id = connectionNoteId(conn->getConnectedToModuleIds());
      auto noteIter = notes.notes.find(id);
      if (noteIter != notes.notes.end())
      {
        auto noteXML = noteIter->second;
        Note note(QString::fromStdString(noteXML.noteHTML), QString::fromStdString(noteXML.noteText), noteXML.fontSize, noteXML.position);
        conn->updateNoteFromFile(note);
      }
    }
  }
}

void NetworkEditor::updateDisabledComponents(const DisabledComponents& disabled)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto conn = dynamic_cast<ConnectionLine*>(item))
    {
      auto id = connectionNoteId(conn->getConnectedToModuleIds());
      if (std::find(disabled.disabledConnections.begin(), disabled.disabledConnections.end(), id) != disabled.disabledConnections.end())
      {
        conn->setDisabled(true);
      }
    }

    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      if (std::find(disabled.disabledModules.begin(), disabled.disabledModules.end(), w->getModuleWidget()->getModuleId()) != disabled.disabledModules.end())
      {
        w->getModuleWidget()->setExecutionDisabled(true);
      }
    }
  }
}

void NetworkEditor::executeAll()
{
  preexecute_();
  // explicit type needed for older Qt and/or clang
  std::function<void()> exec = [this]() { controller_->executeAll(*this); };
  QtConcurrent::run(exec);

  //TODO: not sure about this right now.
  //Q_EMIT modified();
  Q_EMIT networkExecuted();
}

void NetworkEditor::executeModule(const ModuleHandle& module, bool fromButton)
{
  preexecute_();
  // explicit type needed for older Qt and/or clang
  std::function<void()> exec = [this, &module, fromButton]() { controller_->executeModule(module, *this, fromButton); };
  QtConcurrent::run(exec);
  //TODO: not sure about this right now.
  //Q_EMIT modified();
  Q_EMIT networkExecuted();
}

ExecutableObject* NetworkEditor::lookupExecutable(const ModuleId& id) const
{
  auto widget = findById(scene_->items(), id.id_);
  return widget ? widget->getModuleWidget() : 0;
}

void NetworkEditor::resetNetworkDueToCycle()
{
  Q_EMIT resetModulesDueToCycle();
  //TODO: ??reset module colors--right now they stay yellow
}

void NetworkEditor::removeModuleWidget(const ModuleId& id)
{
  auto widget = findById(scene_->items(), id.id_);
  if (widget)
  {
    widget->getModuleWidget()->setDeletedFromGui(false);
    delete widget;
    Q_EMIT modified();
  }
}

void NetworkEditor::clear()
{
  tagLabelOverrides_.clear();
  ModuleWidget::NetworkClearingScope clearing;
  //auto portSwitch = createDynamicPortDisabler();
  scene_->clear();
  //TODO: this (unwritten) method does not need to be called here.  the dtors of all the module widgets get called when the scene_ is cleared, which triggered removal from the underlying network.
  // we'll need a similar hook when programming the scripting interface (moduleWidgets<->modules).
  //controller_->clear();
  Q_EMIT modified();
}

NetworkFileHandle NetworkEditor::saveNetwork() const
{
  return controller_->saveNetwork();
}

void NetworkEditor::loadNetwork(const NetworkFileHandle& xml)
{
  fileLoading_ = true;
  controller_->loadNetwork(xml);
  fileLoading_ = false;

  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
    {
      w->getModuleWidget()->postLoadAction();
    }
  }

  setSceneRect(QRectF());
}

void NetworkEditor::appendToNetwork(const NetworkFileHandle& xml)
{
  auto originalItems = scene_->items();
  fileLoading_ = true;
  controller_->appendToNetwork(xml);
  fileLoading_ = false;

  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (!originalItems.contains(item))
    {
      if (auto w = dynamic_cast<ModuleProxyWidget*>(item))
      {
        w->getModuleWidget()->postLoadAction();
        w->setSelected(true);
      }
    }
  }

  setSceneRect(QRectF());
}

void NetworkEditor::disableViewScenes()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto c = dynamic_cast<ConnectionLine*>(item))
    {
      if (c->id().id_.find("ViewScene") != std::string::npos)
        c->setDisabled(true);
    }
  }
  //TODO: doesn't work yet.
  //Application::Instance().controller()->connectNetworkExecutionFinished([this](int code){ enableViewScenes(); });
}

void NetworkEditor::enableViewScenes()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto c = dynamic_cast<ConnectionLine*>(item))
    {
      if (c->id().id_.find("ViewScene") != std::string::npos)
        c->setDisabled(false);
    }
  }
}

size_t NetworkEditor::numModules() const
{
  return controller_->numModules();
}

void NetworkEditor::setConnectionPipelineType(int type)
{
  ModuleWidget::connectionFactory_->setType(ConnectionDrawType(type));
}

int NetworkEditor::connectionPipelineType() const
{
  return static_cast<int>(ModuleWidget::connectionFactory_->getType());
}

int NetworkEditor::errorCode() const
{
  return controller_->errorCode();
}

ModuleEventProxy::ModuleEventProxy()
{
  qRegisterMetaType<std::string>("std::string");
  qRegisterMetaType<SCIRun::Dataflow::Networks::ModuleHandle>("SCIRun::Dataflow::Networks::ModuleHandle");
  qRegisterMetaType<SCIRun::Dataflow::Networks::ConnectionDescription>("SCIRun::Dataflow::Networks::ConnectionDescription");
  qRegisterMetaType<SCIRun::Dataflow::Networks::ModuleId>("SCIRun::Dataflow::Networks::ModuleId");
  qRegisterMetaType<SCIRun::Dataflow::Networks::ConnectionId>("SCIRun::Dataflow::Networks::ConnectionId");
  qRegisterMetaType<SCIRun::Dataflow::Engine::ModuleCounter>("SCIRun::Dataflow::Engine::ModuleCounter");
}

void ModuleEventProxy::trackModule(ModuleHandle module)
{
  module->connectExecuteBegins([this](const std::string& id) { moduleExecuteStart(id); });
  module->connectExecuteEnds([this](double t, const std::string& id) { moduleExecuteEnd(t, id); });
}

void NetworkEditor::disableInputWidgets()
{
  //deleteAction_->setDisabled(true);
}

void NetworkEditor::enableInputWidgets()
{
  //deleteAction_->setEnabled(true);
}

void NetworkEditor::setBackground(const QBrush& brush)
{
  scene_->setBackgroundBrush(brush);
  Preferences::Instance().networkBackgroundColor.setValue(brush.color().name().toStdString());
}

QBrush NetworkEditor::background() const
{
  return scene_->backgroundBrush();
}

QPixmap NetworkEditor::sceneGrab()
{
  //TODO: this approach may not be able to show the hidden parts of the network.
  return QPixmap::grabWidget(this);
}

void NetworkEditor::selectAll()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    item->setSelected(true);
  }
}

void NetworkEditor::pinAllModuleUIs()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto module = getModule(item);
    if (module)
      module->pinUI();
  }
}

void NetworkEditor::hideAllModuleUIs()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto module = getModule(item);
    if (module)
      module->hideUI();
  }
}

void NetworkEditor::restoreAllModuleUIs()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto module = getModule(item);
    if (module)
      module->showUI();
  }
}

namespace
{
  const double minScale = 0.03;
  const double maxScale = 4.0;
  const double scaleFactor = 1.15;
}

void NetworkEditor::wheelEvent(QWheelEvent* event)
{
  if (event->modifiers() & Qt::ShiftModifier)
  {
    setTransformationAnchor(AnchorUnderMouse);

    if (event->delta() > 0)
    {
      zoomIn();
    }
    else
    {
      zoomOut();
    }
    // Don't call superclass handler here
    // as wheel is normally used for moving scrollbars
  }
  else
    QGraphicsView::wheelEvent(event);
}

void NetworkEditor::zoomIn()
{
  if (currentScale_ < maxScale)
  {
    double factor = std::min(scaleFactor, 4.0/currentScale_);
    scale(factor, factor);
    currentScale_ *= factor;
    Q_EMIT zoomLevelChanged(currentZoomPercentage());
  }
}

void NetworkEditor::zoomOut()
{
  if (currentScale_ > minScale)
  {
    scale(1.0 / scaleFactor, 1.0 / scaleFactor);
    currentScale_ /= scaleFactor;
    Q_EMIT zoomLevelChanged(currentZoomPercentage());
  }
}

void NetworkEditor::zoomReset()
{
  scale(1.0 / currentScale_, 1.0 / currentScale_);
  currentScale_ = 1;
  Q_EMIT zoomLevelChanged(currentZoomPercentage());
}

void NetworkEditor::zoomBestFit()
{
  //TODO not quite working yet.
  auto oldRect = sceneRect();
  setSceneRect(QRectF());
  fitInView(sceneRect(), Qt::KeepAspectRatio);
  //qDebug() << "old rect: " << oldRect << "new rect:" << sceneRect();
  currentScale_ *= sceneRect().x() / oldRect.x();
  //scale(1.0 / currentScale_, 1.0 / currentScale_);
  //currentScale_ = 1;
  Q_EMIT zoomLevelChanged(currentZoomPercentage());
}

int NetworkEditor::currentZoomPercentage() const
{
  return static_cast<int>(currentScale_ * 100);
}

bool NetworkEditor::containsViewScene() const
{
  return findFirstByName(scene_->items(), "ViewScene") != nullptr;
}

void NetworkEditor::moduleWindowAction()
{
  auto action = qobject_cast<QAction*>(sender());
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto module = getModule(item);
    if (module && module->getModuleId() == action->text().toStdString())
    {
      if (module->guiVisible())
        module->hideUI();
      else
        module->showUI();
      break;
    }
  }
}

void NetworkEditor::adjustModuleWidth(int delta)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto proxy = getModuleProxy(item);
    if (proxy)
    {
      proxy->adjustWidth(delta);
      //qDebug() << module->size() << proxy->minimumSize() << proxy->maximumSize() << proxy->preferredSize();
    }
  }
}

void NetworkEditor::adjustModuleHeight(int delta)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto proxy = getModuleProxy(item);
    if (proxy)
    {
      proxy->adjustHeight(delta);
      //qDebug() << module->size() << proxy->minimumSize() << proxy->maximumSize() << proxy->preferredSize();
    }
  }
}

void NetworkEditor::metadataLayer(bool active)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    item->setOpacity(active ? 0.4 : 1);
    auto module = getModule(item);
    if (module)
      module->updateMetadata(active);
  }
}

void NetworkEditor::adjustExecuteButtonsToDownstream(bool downOnly)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto module = getModule(item);
    if (module)
    {
      module->adjustExecuteButtonToDownstream(downOnly);
    }
  }
}

QColor Gui::defaultTagColor(int tag)
{
  switch (tag)
  {
  case 0:
    return Qt::blue;
  case 1:
    return Qt::green;
  case 2:
    return Qt::darkYellow;
  case 3:
    return Qt::darkMagenta;
  case 4:
    return Qt::darkCyan;
  case 5:
    return Qt::darkRed;
  case 6:
    return Qt::darkGray;
  case 7:
    return Qt::darkGreen;
  case 8:
    return Qt::darkBlue;
  case 9:
    return Qt::black;
  default:
    return Qt::white;
  }
}

QString Gui::colorToString(const QColor& color)
{
  return QString("rgb(%1, %2, %3)").arg(color.red()).arg(color.green()).arg(color.blue());
}

QGraphicsEffect* Gui::blurEffect(double radius)
{
  auto blur = new QGraphicsBlurEffect;
  blur->setBlurRadius(radius);
  return blur;
}

void NetworkEditor::tagLayer(bool active, int tag)
{
  tagLayerActive_ = active;

  if (active)
  {
    auto items = scene_->selectedItems();
    Q_FOREACH(QGraphicsItem* item, items)
    {
      if (item->data(TagDataKey).toInt() == NoTag)
      {
        if (validTag(tag))
          item->setData(TagDataKey, tag);
      }
      else if (ClearTags == tag)
      {
        item->setData(TagDataKey, NoTag);
      }
    }
  }

  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    item->setData(TagLayerKey, active);
    item->setData(CurrentTagKey, tag);
    if (active)
    {
      const auto itemTag = item->data(TagDataKey).toInt();
      if (AllTags == tag || ShowGroups == tag)
      {
        highlightTaggedItem(item, itemTag);
      }
      else if (tag != NoTag && tag != ClearTags)
      {
        if (tag == itemTag)
        {
          highlightTaggedItem(item, itemTag);
        }
        else
          item->setGraphicsEffect(blurEffect());
      }
    }
    else
      item->setGraphicsEffect(nullptr);
  }
  if (ShowGroups == tag)
  {
    tagGroupsActive_ = true;
    redrawTagGroups();
  }
  if (HideGroups == tag)
  {
    tagGroupsActive_ = false;
    removeTagGroups();
  }
}

namespace
{
  class TagGroupBox : public QGraphicsRectItem
  {
  public:
    explicit TagGroupBox(int tagNum, const QRectF& rect, NetworkEditor* ned) : QGraphicsRectItem(rect),
      tagNumber_(tagNum),
      ned_(ned)
    {
      setAcceptHoverEvents(true);
    }
  protected:
    virtual void hoverEnterEvent(QGraphicsSceneHoverEvent*) override
    {
      setPen(QPen(pen().color(), 5));
    }

    virtual void hoverLeaveEvent(QGraphicsSceneHoverEvent*) override
    {
      setPen(QPen(pen().color(), 3));
    }

    virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) override
    {
      QMenu menu;
      auto autoDisplay = menu.addAction("Display in saved network", ned_, SLOT(saveTagGroupRectInFile()));
      autoDisplay->setCheckable(true);
      autoDisplay->setChecked(ned_->showTagGroupsOnFileLoad());
      auto rename = menu.addAction("Rename in saved network...", ned_, SLOT(renameTagGroupInFile()));
      rename->setProperty("tag", tagNumber_);
      menu.exec(event->screenPos());
      QGraphicsRectItem::mouseDoubleClickEvent(event);
    }
  private:
    int tagNumber_;
    NetworkEditor* ned_;
  };
}

void NetworkEditor::saveTagGroupRectInFile()
{
  auto action = qobject_cast<QAction*>(sender());
  setShowTagGroupsOnFileLoad(action->isChecked());
  Q_EMIT modified();
}

void NetworkEditor::renameTagGroupInFile()
{
  auto action = qobject_cast<QAction*>(sender());
  auto tagNum = action->property("tag").toInt();

  bool ok;
  auto text = QInputDialog::getText(this, tr("Rename tag group"),
    tr("Enter new tag group name for this network file:"), QLineEdit::Normal, checkForOverriddenTagName(tagNum), &ok);
  if (ok && !text.isEmpty())
  {
    bool changed = tagLabelOverrides_[tagNum] != text.toStdString();
    tagLabelOverrides_[tagNum] = text.toStdString();
    if (changed)
      renameTagGroup(tagNum, text);
  }

  Q_EMIT modified();
}

class SubnetworkGraphicsItem : public QGraphicsPixmapItem
{
public:
  explicit SubnetworkGraphicsItem(const QPixmap& pic) : QGraphicsPixmapItem(pic) {}

  virtual void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget) override
  {
    painter->setPen(QPen(QColor(220,220,220,80), 18.0, Qt::SolidLine));
    painter->drawRoundedRect(boundingRect(), 15, 15);

    QGraphicsPixmapItem::paint(painter, option, widget);
  }
};

class SubnetModule : public Module
{
public:
  explicit SubnetModule(const std::vector<ModuleHandle>& underlyingModules) : Module(ModuleLookupInfo()),
    underlyingModules_(underlyingModules)
  {}
  virtual void execute() override
  {
    //std::ostringstream ostr;
    //ostr << "Module " << get_module_name() << " executing for " << 3.14 << " seconds." << std::endl;
    //status(ostr.str());
  }
  virtual void setStateDefaults() override
  {}
  std::string listComponentIds() const
  {
    std::ostringstream ostr;
    std::transform(underlyingModules_.begin(), underlyingModules_.end(),
      std::ostream_iterator<std::string>(ostr, ", "),
      [](const ModuleHandle& mod) { return mod->get_id(); });
    return ostr.str();
  }
private:
  std::vector<ModuleHandle> underlyingModules_;
};

void NetworkEditor::makeSubnetwork()
{
  QRectF rect;
  QPointF position;

  std::vector<ModuleHandle> underlyingModules;
  Q_FOREACH(QGraphicsItem* item, scene_->selectedItems())
  {
    auto r = item->boundingRect();
    position = item->pos();
    r = item->mapRectToParent(r);

    if (rect.isEmpty())
      rect = r;
    else
      rect = rect.united(r);

    auto module = getModule(item);
    if (module)
      underlyingModules.push_back(module->getModule());
  }
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (dynamic_cast<QGraphicsPixmapItem*>(item))
      item->setVisible(false);
  }

  auto pic = QPixmap::grabWidget(this, mapFromScene(rect).boundingRect());

  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (dynamic_cast<QGraphicsPixmapItem*>(item))
      item->setVisible(true);
  }
  // auto picItem = new SubnetworkGraphicsItem(pic);
  // picItem->setFlags(QGraphicsItem::ItemIsMovable | QGraphicsItem::ItemIsSelectable);
  //
  // auto picRect = picItem->boundingRect();
  // auto longEdge = std::max(picRect.height(), picRect.width());
  // picItem->setScale(100.0 / longEdge);
  //
  // scene_->addItem(picItem);
  //
  // while (!scene_->items(position.x() - 20, position.y() - 20, 40, 40).isEmpty())
  // {
  //   position += QPointF(100, 0);
  // }
  // picItem->setPos(position);
  // scene_->clearSelection();
  // picItem->setSelected(true);

  auto name = QInputDialog::getText(nullptr, "Make subnet", "Enter subnet name:");

  auto subnetModule = boost::make_shared<SubnetModule>(underlyingModules);
  std::cout << "Subnet components: " << subnetModule->listComponentIds() << std::endl;
  auto moduleWidget = new ModuleWidget(this, "Subnet::" + name, subnetModule, dialogErrorControl_);


  QByteArray byteArray;
  QBuffer buffer(&byteArray);
  pic.scaled(pic.size() * 0.5).save(&buffer, "PNG");
  QString tooltipPic = QString("<html><img src=\"data:image/png;base64,") + byteArray.toBase64() + "\"/></html>";

  auto proxy = setupModuleWidget(moduleWidget);
  proxy->setScale(1.5);
  proxy->setToolTip(tooltipPic);
}

void NetworkEditor::drawTagGroups()
{
  if (tagGroupsActive_)
  {
    QMap<int, QRectF> tagItemRects;

    Q_FOREACH(QGraphicsItem* item, scene_->items())
    {
      if (dynamic_cast<ModuleProxyWidget*>(item))
      {
        const auto itemTag = item->data(TagDataKey).toInt();

        if (itemTag != NoTag)
        {
          auto r = item->boundingRect();
          r.translate(item->pos());
          if (!tagItemRects.contains(itemTag))
          {
            tagItemRects.insert(itemTag, r);
          }
          else
          {
            tagItemRects[itemTag] = tagItemRects[itemTag].united(r);
          }
        }
      }
    }

    for (auto rectIter = tagItemRects.constBegin(); rectIter != tagItemRects.constEnd(); ++rectIter)
    {
      auto rectBounds = rectIter.value().adjusted(-10, -10, 10, 10);
      auto tagNum = rectIter.key();
      QPen pen(tagColor_(tagNum));
      pen.setWidth(3);
      pen.setCapStyle(Qt::RoundCap);
      pen.setJoinStyle(Qt::RoundJoin);
      auto rect = new TagGroupBox(tagNum, rectBounds, this);
      rect->setPen(pen);
      scene_->addItem(rect);
      rect->setFlags(QGraphicsItem::ItemIsFocusable | QGraphicsItem::ItemIsSelectable);
      rect->setZValue(-100000);

      auto fill = new QGraphicsRectItem(rectBounds);
      auto c = pen.color();
      c.setAlphaF(0.15);
      fill->setBrush(c);
      fill->setZValue(-100000);
      scene_->addItem(fill);

      static const QFont labelFont("Courier", 20, QFont::Bold);

      auto label = scene_->addSimpleText(checkForOverriddenTagName(tagNum), labelFont);
      label->setBrush(pen.color());
      label->setData(TagTextKey, tagNum);
      static const QFontMetrics fm(labelFont);
      auto textWidthInPixels = fm.width(label->text());
      label->setPos((rect->rect().topLeft() + rect->rect().topRight()) / 2 + QPointF(-textWidthInPixels / 2, -30));
    }
  }
}

QString NetworkEditor::checkForOverriddenTagName(int tag) const
{
  auto nameOverrideIter = tagLabelOverrides_.find(tag);
  if (nameOverrideIter != tagLabelOverrides_.end() && !nameOverrideIter->second.empty())
    return QString::fromStdString(nameOverrideIter->second);
  return tagName_(tag);
}

void NetworkEditor::removeTagGroups()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (dynamic_cast<QGraphicsRectItem*>(item) || dynamic_cast<QGraphicsSimpleTextItem*>(item))
    {
      delete item;
    }
  }
}

void NetworkEditor::renameTagGroup(int tag, const QString& name)
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    if (auto rectLabel = dynamic_cast<QGraphicsSimpleTextItem*>(item))
    {
      if (rectLabel->data(TagTextKey).toInt() == tag)
      {
        rectLabel->setText(name);
        return;
      }
    }
  }
}

void NetworkEditor::redrawTagGroups()
{
  removeTagGroups();
  drawTagGroups();
}

void NetworkEditor::highlightTaggedItem(int tagValue)
{
  highlightTaggedItem(qobject_cast<QGraphicsItem*>(sender()), tagValue);
  Q_EMIT modified();
}

void NetworkEditor::highlightTaggedItem(QGraphicsItem* item, int tagValue)
{
  if (tagValue == NoTag)
  {
    item->setGraphicsEffect(blurEffect());
  }
  else
  {
    auto colorize = new QGraphicsColorizeEffect;
    auto color = tagColor_(tagValue);
    colorize->setColor(color);
    item->setGraphicsEffect(colorize);
  }
}

void NetworkEditor::cleanUpNetwork()
{
  controller_->cleanUpNetwork();
  centerView();
}

ErrorItem::ErrorItem(const QString& text, std::function<void()> action, QGraphicsItem* parent) : FloatingTextItem(text, action, parent)
{
  setDefaultTextColor(Qt::red);
}

std::atomic<int> FloatingTextItem::instanceCounter_(0);

FloatingTextItem::FloatingTextItem(const QString& text, std::function<void()> action, QGraphicsItem* parent) : QGraphicsTextItem(text, parent),
  action_(action), counter_(instanceCounter_), rect_(nullptr)
{
  setFlags(ItemIsMovable | ItemIsSelectable | ItemSendsGeometryChanges);
  setZValue(10000);
  ++instanceCounter_;

  {
    timeLine_ = new QTimeLine(10000, this);
    connect(timeLine_, SIGNAL(valueChanged(qreal)), this, SLOT(animate(qreal)));
    connect(timeLine_, SIGNAL(finished()), this, SLOT(deleteLater()));
  }
  timeLine_->start();
}

FloatingTextItem::~FloatingTextItem()
{
  --instanceCounter_;
  delete rect_;
}

void FloatingTextItem::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
  if (event->buttons() & Qt::LeftButton)
  {
    if (action_)
      action_();
  }
  else if (event->buttons() & Qt::RightButton)
  {
    if (rect_)
    {
      scene()->removeItem(rect_);
      rect_ = nullptr;
    }
    scene()->removeItem(this);
  }
  QGraphicsTextItem::mousePressEvent(event);
}

void FloatingTextItem::hoverEnterEvent(QGraphicsSceneHoverEvent *event)
{
  timeLine_->setCurrentTime(0);
  QGraphicsTextItem::hoverEnterEvent(event);
}

void FloatingTextItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent *event)
{
  if (!rect_)
  {
    timeLine_->stop();
    timeLine_->setCurrentTime(0);
    auto f = font();
    f.setBold(true);
    setFont(f);
    setFlags(flags() & ~ItemIsMovable);
    rect_ = scene()->addRect(boundingRect(), QPen(defaultTextColor(), 2, Qt::DotLine));
    rect_->setPos(pos());
  }
  else
  {
    auto f = font();
    f.setBold(false);
    setFont(f);
    setFlags(flags() & ItemIsMovable);
    scene()->removeItem(rect_);
    rect_ = nullptr;
    timeLine_->start();
  }

  QGraphicsTextItem::mouseDoubleClickEvent(event);
}

void FloatingTextItem::animate(qreal val)
{
  if (val < 1)
    show();
  else
    hide();
  setOpacity(val < 0.5 ? 1 : 2 - 2*val);
}

NetworkEditor::~NetworkEditor()
{
  Q_FOREACH(QGraphicsItem* item, scene_->items())
  {
    auto module = getModule(item);
    if (module)
      module->setDeletedFromGui(false);
  }
  NetworkEditor::clear();
}

ZLevelManager::ZLevelManager(QGraphicsScene* scene)
  : scene_(scene), minZ_(INITIAL_Z), maxZ_(INITIAL_Z)
{

}
