import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, BatchSampler, Dataset
from skimage.metrics import structural_similarity as ssim
# from pytorch_ssim._init_ import ssim, ms_ssim, SSIM, MS_SSIM
dtype = torch.cuda.FloatTensor
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        combined_conv = F.relu(combined_conv)  # Using ReLU instead of ELU

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        device = next(self.parameters()).device
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))


class ConvEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, dropout_prob=0.5):
        super(ConvEncoder, self).__init__()
        self.conv_lstm1 = ConvLSTMCell(input_channels, hidden_channels , kernel_size)
        self.conv_lstm2 = ConvLSTMCell(hidden_channels , hidden_channels // 2, kernel_size)
        self.batch_norm = nn.BatchNorm2d(hidden_channels // 2)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, input_tensor):
        batch_size, seq_len, _, height, width = input_tensor.size()
        h1, c1 = self.conv_lstm1.init_hidden(batch_size, (height, width))
        h2, c2 = self.conv_lstm2.init_hidden(batch_size, (height, width))
        last_state = None

        for t in range(seq_len):
            h1, c1 = self.conv_lstm1(input_tensor[:, t, :, :, :], (h1, c1))
            h2, c2 = self.conv_lstm2(h1, (h2, c2))

            h2 = self.batch_norm(h2)
            h2 = self.dropout(h2)
            if t == seq_len - 1:
                last_state = h2

        return last_state, (h2, c2)

class ConvDecoder(nn.Module):
    def __init__(self, hidden_channels, output_channels, kernel_size, dropout_prob=0.5):
        super(ConvDecoder, self).__init__()
        self.conv_lstm1 = ConvLSTMCell(hidden_channels //2, hidden_channels //2 , kernel_size)
        self.conv_lstm2 = ConvLSTMCell(hidden_channels //2 , hidden_channels // 2, kernel_size)


        self.batch_norm = nn.BatchNorm2d(hidden_channels // 2)
        self.conv = nn.Conv2d(hidden_channels // 2, output_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, encoder_last_state, h, c, seq_len):
        outputs = []

        for t in range(seq_len):
            h, c = self.conv_lstm1(h, (h, c))
            h, c = self.conv_lstm2(h, (h, c))
            h = self.batch_norm(h)
            if t == 0:
                h = h + encoder_last_state
            h = self.dropout(h)
            outputs.append(h)

        output = self.conv(outputs[-1])
        output = output.unsqueeze(1)

        return output

class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, kernel_size):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = ConvEncoder(input_channels, hidden_channels, kernel_size)
        self.decoder = ConvDecoder(hidden_channels, output_channels, kernel_size)

    def forward(self, input_tensor):
        encoder_last_state, (h, c) = self.encoder(input_tensor)
        output = self.decoder(encoder_last_state, h, c, seq_len=1)
        return output

# Model initialization
input_channels = 1
hidden_channels = 128
output_channels = 1
kernel_size = 1
sequence_length = 5

model = Seq2SeqAutoencoder(input_channels, hidden_channels, output_channels, kernel_size)

# Test the model
input_tensor = torch.rand(1, sequence_length, input_channels, 224, 224)
output = model(input_tensor)
print(output.shape)  # Check the output shape