from torch.utils.data import DataLoader
import torch
import unittest

from model import Network as Network
#https://krokotsch.eu/posts/deep-learning-unit-tests/
#%%
class My_unittest(unittest.TestCase):
    def __init__(self,net):
        self.net=net()
    
    def setUp(self):

        

        #define test input data here
        self.test_input = torch.randn(4,1,32,32)
    """
    Model test
    """
    #check output shape
    @torch.no_grad()
    def test_shape(self):
        outputs = self.net(self.test_input)
        self.assertEqual(self.test_input.shape, outputs.shape)

    #check if data is moveable to gpu
    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_device_moving(self):
        net_on_gpu = self.net.to('cuda:0')
        net_back_on_cpu = net_on_gpu.cpu()
        


        torch.manual_seed(42)
        outputs_cpu = self.net(self.test_input)
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(self.test_input.to('cuda:0'))
        torch.manual_seed(42)
        outputs_back_on_cpu = net_back_on_cpu(self.test_input)

        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()))
        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_back_on_cpu))
