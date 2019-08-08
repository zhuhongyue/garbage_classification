# garbage_classification
A simple vgg16 classification model (based Imagnet pretrained model) in pytorch.
The validation accuracy durning training  is 0.8.
## data
you can collect by yourself in search engine

## model
after get dataset you can run train.py to get model file named best_checkpoint.pth.tar in ./checkpoint/

## demo
after get dataset and model,you can run demo.py to test performance

## deployment
the file in ./deployment can be used for deployment,run deployment.py for boot server and request.py for test 
PS. request.py is not maintined any more, you can use postman(easy to find download address in google.com) to test the performance

## Note
all the path value in code file may cause erro, so please watch out the file path setting.
or maybe you can add parameter feature for setting path (The author (myself) left nothing with laziness abou this)

