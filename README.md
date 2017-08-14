# finetune_vggface_morph
Idea from  kratzert/finetune_alexnet_with_tensorflow and fellow the main route to finetune Vgg-Face on Morph dataset  for age estimation.
When finish finetune. Can get MAE 5.5 for age estimation.

Main Finetune Steps:
1. Download the Caffe pretrained model, (if the model is old version, use Caffe util (upgrade_net_proto_binary   upgrade_net_proto_text ) to upgrade the model file).
2. Clone the git caffe-tensorflow and use fellow command to generate tensorflow network file and numpy binary dump file.
    ./convert.py examples/mnist/lenet.prototxt --code-output-path=mynet.py
    ./convert.py examples/mnist/lenet.prototxt --caffemodel examples/mnist/lenet_iter_10000.caffemodel --data-output-path=mynet.npy

3. Write you own tensorflow finetune file, and load the weights from the numpy dumped file. 
4. You should better validate it  before finetune you own data  too make sure everything is ok


