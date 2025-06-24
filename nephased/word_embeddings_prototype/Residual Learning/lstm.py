import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import torch.jit as jit

use_cuda = torch.cuda.is_available()

class ResLSTMCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMCell, self).__init__()
        self.register_buffer('input_size', torch.Tensor([input_size]))
        self.register_buffer('hidden_size', torch.Tensor([hidden_size]))

        # A single weight matrix for all 4 gates from the input
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))

        # A single weight matrix for all 4 gates from the hidden state

        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))

        # A single bias vector for all 4 gates
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))

        # Residual projection 
        self.weight_ir = nn.Parameter(torch.randn(hidden_size, input_size))

        # LayerNorms for the residual connection
        self.ln_ry = nn.LayerNorm(hidden_size)
        self.ln_res = nn.LayerNorm(hidden_size)

    @jit.script_method
    def forward(self, this_input, hidden):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        hx, cx = hidden[0].squeeze(0), hidden[1].squeeze(0)

        gates = torch.matmul(this_input, self.weight_ih.t()) + torch.matmul(hx, self.weight_hh.t()) + self.bias

        ingate, forgetgate,cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        ry = torch.tanh(cy)

        ry_normalized = self.ln_ry(ry)

        if self.input_size == self.hidden_size:
            res_connection = self.ln_res(this_input)
            hy = outgate * (ry_normalized + res_connection)
        else:
            projected_input = torch.matmul(this_input, self.weight_ir.t())
            res_connection = self.ln_res(projected_input)
            hy = outgate * (ry_normalized + res_connection)
        return hy, (hy, cy)

class ResLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.):
        super(ResLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ResLSTMCell(input_size, hidden_size, dropout=0.)

    def forward(self, this_input, hidden):
        inputs = this_input.unbind(0)
        
        outputs = []
        for i in range(len(inputs)):
            # output hidden state at each time step is passed onto the next time step
            cell_result = self.cell(inputs[i], hidden)
            out, hidden = cell_result[0], cell_result[1]
            outputs.append(out)

        outputs = torch.stack(outputs)
        return outputs, hidden

class ResLSTM(nn.Module):
    """
    A multi-layer Residual LSTM that can be confiNgured to be bidirectional
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0, batch_first=False, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)

        self.forward_cells = nn.ModuleList()
        if self.bidirectional:
            self.backward_cells = nn.ModuleList()

        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_dim
            
            self.forward_cells.append(ResLSTMLayer(input_size, hidden_dim, dropout=dropout))
            if self.bidirectional:
                self.backward_cells.append(ResLSTMLayer(input_size, hidden_dim, dropout=dropout))

        # Call the custom weight initializer 
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights with Xavier Uniform and biases to zero.
        This is a robust starting point now that the cell architecture is stable.
        """
        for name, param in self.named_parameters():
            # Initialize all 2D weight matrices
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Initialize all biases to zero
                nn.init.constant_(param, 0.0)
                
                # The forget gate is the second quarter of the bias vector
                # (since gates are ordered i, f, c, o)
                n = param.size(0)
                if n % 4 == 0:
                    start, end = n // 4, n // 2
                
                    # Setting the forget gate biases to 1.0 to encourage remembering
                    with torch.no_grad():
                        param[start:end].fill_(1.0)

    def forward(self, inputs, hidden=None):
        """
        The forward pass for the ResLSTM.
        Returns:
            output (Tensor): The output sequence from the last layer.
            (h_n, c_n) (Tuple[Tensor, Tensor]): The final hidden and cell states for all layers.
        """
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        # --- Forward Pass ---
        fwd_hidden_states, fwd_cell_states = [], []
        fwd_outputs = inputs
        
        for i, cell in enumerate(self.forward_cells):
            fwd_outputs, (h, c) = self.res_layer(cell, fwd_outputs, hidden=None)
            fwd_hidden_states.append(h.unsqueeze(0))
            fwd_cell_states.append(c.unsqueeze(0))

            if i < self.num_layers - 1:
                fwd_outputs = self.dropout_layer(fwd_outputs)

        h_n = None
        c_n = None

        # If not bidirectional, we're done
        if not self.bidirectional:
            output = fwd_outputs
            h_n = torch.cat(fwd_hidden_states, dim=0)
            c_n = torch.cat(fwd_cell_states, dim=0)
        else:
            # --- Backward Pass ---
            # Reverse the original input sequence for the backward stack
            bwd_hidden_states, bwd_cell_states = [], []
            
            bwd_outputs = torch.flip(inputs, [0])
            for i, cell in enumerate(self.backward_cells):
                bwd_outputs, (h, c) = self.res_layer(cell, bwd_outputs, hidden=None)
                bwd_hidden_states.append(h.unsqueeze(0))
                bwd_cell_states.append(c.unsqueeze(0))

                if i < self.num_layers - 1:
                    bwd_outputs = self.dropout_layer(bwd_outputs)
            
            # Flip the final backward output back to the original time order
            bwd_outputs = torch.flip(bwd_outputs, [0])
            
            # --- Concatenate Final Outputs ---
            output = torch.cat([fwd_outputs, bwd_outputs], dim=2)

            h_n_list = []
            c_n_list = []
            for i in range(self.num_layers):
                h_n_list.append(fwd_hidden_states[i])
                h_n_list.append(bwd_hidden_states[i])
                c_n_list.append(fwd_cell_states[i])
                c_n_list.append(bwd_cell_states[i])
            
            h_n = torch.cat(h_n_list, dim=0)
            c_n = torch.cat(c_n_list, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, (h_n, c_n)

    # The helper methods 
    def res_layer(self, cell, inputs, hidden=None):
        batch_size = inputs.size(1)
        hidden_size = cell.hidden_size
        if hidden is None:
            h, c = self.init_hidden(batch_size, hidden_size)
            hidden = (h.unsqueeze(0), c.unsqueeze(0))
        outputs, hidden = cell(inputs, hidden)
        return outputs, hidden

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim)
        memory = torch.zeros(batch_size, hidden_dim)
        if use_cuda:
            hidden = hidden.cuda()
            memory = memory.cuda()
        return hidden, memory