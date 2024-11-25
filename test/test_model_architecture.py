import unittest
import torch
import torch.nn.functional as F
from torch.nn import Module

# Import your blocks here
from pytorch_mlp_framework.model_architectures import *

class TestBlocks(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.input_shape = (4, 3, 32, 32)  # Batch size 4, 3 channels, 32x32 resolution
        self.num_filters = 16
        self.kernel_size = 3
        self.padding = 1
        self.bias = True
        self.dilation = 1
        self.reduction_factor = 2

    def test_empty_block(self):
        block = EmptyBlock(input_shape=self.input_shape)
        x = torch.randn(self.input_shape)
        output = block(x)
        self.assertEqual(output.shape, x.shape)  # Check that output shape matches input shape

    def test_entry_convolutional_block(self):
        block = EntryConvolutionalBlock(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation
        )
        x = torch.randn(self.input_shape)
        output = block(x)
        print('output: ', output.shape[2:])
        print('input: ', x.shape[2:])
        self.assertEqual(output.shape[0], x.shape[0])  # Batch size remains the same
        self.assertEqual(output.shape[1], self.num_filters)  # Check number of output channels
        self.assertEqual(output.shape[2:], x.shape[2:])  # Check spatial dimensions

    def test_convolutional_processing_block(self):
        block = ConvolutionalProcessingBlock(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation
        )
        x = torch.randn(self.input_shape)
        output = block(x)
        self.assertEqual(output.shape[0], x.shape[0])  # Batch size remains the same
        self.assertEqual(output.shape[1], self.num_filters)  # Check number of output channels
        self.assertEqual(output.shape[2:], x.shape[2:])  # Check spatial dimensions

    def test_convolutional_dimensionality_reduction_block(self):
        block = ConvolutionalDimensionalityReductionBlock(
            input_shape=self.input_shape,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
            dilation=self.dilation,
            reduction_factor=self.reduction_factor
        )
        x = torch.randn(self.input_shape)
        output = block(x)
        self.assertEqual(output.shape[0], x.shape[0])  # Batch size remains the same
        self.assertEqual(output.shape[1], self.num_filters)  # Check number of output channels
        reduced_size = x.shape[2] // self.reduction_factor  # pooling的计算化简了
        self.assertEqual(output.shape[2:], (reduced_size, reduced_size))  # Check reduced spatial dimensions

    def test_convolutional_network(self):
        num_stages = 2
        num_blocks_per_stage = 2
        num_output_classes = 10

        net = ConvolutionalNetwork(
            input_shape=self.input_shape,
            num_output_classes=num_output_classes,
            num_filters=self.num_filters,
            num_blocks_per_stage=num_blocks_per_stage,
            num_stages=num_stages,
            use_bias=self.bias
        )
        x = torch.randn(self.input_shape)
        output = net(x)
        self.assertEqual(output.shape[0], x.shape[0])  # Batch size remains the same
        self.assertEqual(output.shape[1], num_output_classes)  # Check number of output classes

if __name__ == '__main__':
    # unittest.main()
    test_block = TestBlocks()
    test_block.setUp()
    # test_block.test_empty_block()
    # test_block.test_entry_convolutional_block()
    test_block.test_convolutional_dimensionality_reduction_block()

