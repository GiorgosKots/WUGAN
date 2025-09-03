import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import gumbel_softmax

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)
    
class Generator_lang(nn.Module):
    def __init__(self, n_chars, seq_len, batch_size, hidden):
        super(Generator_lang, self).__init__()
        self.fc1 = nn.Linear(128, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden = hidden

    def forward(self, noise):
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len, -1)
        output = gumbel_softmax(output, 0.5)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.res_block = ResBlock(in_channels)
        self.down_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )

    def forward(self, input):
        x = self.res_block(input)
        return self.down_block(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding=0):
        super(UpBlock, self).__init__()
        self.res_block = ResBlock(in_channels)
        self.up_block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=output_padding),
            nn.ReLU(inplace=False),
        )

    def forward(self, input):
        x = self.res_block(input)
        return self.up_block(x)

class UNetDiscriminator(nn.Module):
    def __init__(self, n_chars, seq_len, hidden):
        super(UNetDiscriminator, self).__init__()
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden

        # self.conv1d = nn.Conv1d(n_chars, hidden, 1)

        # Downsampling path
        self.down1 = DownBlock(n_chars, hidden)
        self.down2 = DownBlock(hidden, hidden * 2)
        self.down3 = DownBlock(hidden * 2, hidden * 4)
        self.down4 = DownBlock(hidden * 4, hidden * 8)
        self.down5 = DownBlock(hidden * 8, hidden * 16)

        # Bottleneck
        self.bottleneck = ResBlock(hidden * 16)

        # Global output layer
        self.global_output_layer = nn.Linear(hidden * 16 * (seq_len // 32), 1)

        # Upsampling path
        self.up1 = UpBlock(hidden * 16, hidden * 8, output_padding=1)
        self.up2 = UpBlock(hidden * 8, hidden * 4, output_padding=1)
        self.up3 = UpBlock(hidden * 4, hidden * 2, output_padding=1)
        self.up4 = UpBlock(hidden * 2, hidden)
        self.up5 = UpBlock(hidden, n_chars)

        # Pixel-wise output layer
        self.pixel_output_layer = nn.Conv1d(n_chars, 1, kernel_size=1)

    def forward(self, input):
        # Convert to [batch, n_chars, seq_len] for conv1d
        if input.shape[1] == self.seq_len:
            input = input.transpose(1, 2) 
        # x = self.conv1d(input)

        # Downsampling path
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Bottleneck
        b = self.bottleneck(d5)

        # Global output
        global_output = b.view(-1, self.hidden * 16 * (self.seq_len // 32))
        global_output = self.global_output_layer(global_output)

        # Upsampling path
        u1 = self.up1(b)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        u5 = self.up5(u4)

        # Pixel-wise output
        pixel_output = self.pixel_output_layer(u5)
        pixel_output = torch.sigmoid(pixel_output)
        pixel_output = pixel_output.transpose(1, 2) # Back to [batch, seq_len, 1]

        return global_output, pixel_output