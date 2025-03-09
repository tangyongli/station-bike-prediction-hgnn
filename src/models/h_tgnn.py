
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.nn as nn
from torch_geometric.nn import GCNConv



# ==================================================
# Heterogeneous Graph + GRU
# ==================================================
class HeteroGNN(nn.Module):
    def __init__(self, hidden_channels, hidden_units_gru,prediction_length):
        super().__init__()
       
        #--- Pre-process node features---
        self.lin_dict = torch.nn.ModuleDict({
            'station': nn.Sequential(Linear(-1, hidden_channels),nn.ReLU()) ,
            'poi_category': nn.Sequential(Linear(-1, hidden_channels),nn.ReLU()),
            'census_zone': nn.Sequential(Linear(-1, hidden_channels), nn.ReLU())
        })
        # --- Linear layers for projecting features to hidden_channels ---
        # self.lin_flow = nn.Sequential(Linear(-1, hidden_channels),nn.ReLU())
        self.time_projection = nn.Sequential(Linear(-1, hidden_channels,nn.ReLU()))
            
        # --- HeteroConv (using RGCNConv or SAGEConv) ---
        self.conv_static = HeteroConv({
            ('station', 'near_poi_category', 'poi_category'): SAGEConv((-1, -1), hidden_channels),
            ('station', 'located_in', 'census_zone'): SAGEConv((-1, -1), hidden_channels),
            ('station', 'self_loop', 'station'): SAGEConv((-1,-1), hidden_channels),  # The self-loop
        }, aggr='sum')
        
        # --- GRU and Output Layers ---
        ## GRU的输入tensor的通道维度是HeteroConV卷积中的三个edge types输出通道的总和
        self.gru = torch.nn.GRU(input_size=2*hidden_channels, hidden_size=hidden_units_gru, batch_first=True)
        self.lin_final_flow = Linear(hidden_units_gru, prediction_length)  
       
    def forward(self, x_dict):
        # print(x_dict['features'].keys())
        batch_size, seq_length,num_stations, _ = x_dict['features']['station_flows'].shape
        # ---HeteroConv---
        # ---1.STATIC EMBEDDING---
        x_dict_static = {
            'station': x_dict['features']['station_poi'][0], # Static POI features,remove the first demension of batch. Shape: (num_stations, num_poi_features)
            'poi_category': x_dict['features']['poi_category'][0],
            'census_zone':x_dict['features']['census_zone'][0]
        }
        x_dict_static = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict_static.items()
        }
        edge_index_static = {
             ('station', 'near_poi_category', 'poi_category'): x_dict['edge_indices'][('station', 'near_poi_category', 'poi_category')][0],
            ('station', 'located_in', 'census_zone') :x_dict['edge_indices'][('station', 'located_in', 'census_zone')][0]}                                                                   
                                                                                                                                                  
        # 2. ---Add self-loops dynamically---
        # Create edges from each station to itself.
        edge_index_self_loop = torch.arange(0, num_stations,device=x_dict['features']['poi_category'].device)
        edge_index_self_loop = torch.stack([edge_index_self_loop, edge_index_self_loop], dim=0) # (2,num_stations)
        edge_index_static[('station', 'self_loop', 'station')] = edge_index_self_loop

        # 3.---call hetergous convulation---
        out_dict_static = self.conv_static(x_dict_static, edge_index_static)
        # print(out_dict_static.keys())
        out_dict_static = {key: F.dropout(F.relu(x), p=0.2, training=self.training) for key, x in out_dict_static.items()}
        station_embed = out_dict_static['station']  # (num_stations, hidden_channels)
        # print('station_embed',station_embed.shape)
        station_embed=station_embed.unsqueeze(0).unsqueeze(0) #(1,1,num_stations, hidden_channels)
        station_embed = station_embed.expand(batch_size, seq_length, -1,-1) # [batch_size,seq_length,num_stations, hidden_channels] 
        # print(station_embed.shape)
        
        # --- Temporal Processing with GRU ---
        ## 4. Aggregate temporal features
        station_flow_features = x_dict['features']['station_flows']
        hours= x_dict['features']['hour_of_day']
        days=x_dict['features']['day_of_week']
        weather=x_dict['features']['weather']
        time_features=torch.concatenate([station_flow_features,hours,days,weather],axis=-1) #[64, 12, 1807,36]
        # station_flow_features = self.lin_flow(station_flow_features)  # (batch_size, seq_length,num_stations, hidden_channels)
        time_features = self.time_projection(time_features) 
        station_sequences_output=torch.concatenate([time_features,station_embed],axis=-1)
        
        # 5.Reshape for GRU to treat each station's sequence independently - shape (batch_size * num_stations, sequence_length, out_channels)
        station_sequences_output = station_sequences_output.permute(0, 2, 1,3)
        gru_input = station_sequences_output.reshape(station_sequences_output.size(0) * station_sequences_output.size(1), station_sequences_output.size(2), station_sequences_output.size(3)) # (batch_size * num_stations, sequence_length, out_channels)
        gru_out, _ = self.gru(gru_input) # gru_out: (batch_size * num_stations, sequence_length, hidden_units_gru)
        gru_out_last_timestep = gru_out[:, -1, :]
        gru_out_reshaped = gru_out_last_timestep.reshape(batch_size,num_stations,-1)
        output = self.lin_final_flow(gru_out_reshaped)  # (batch_size, num_stations,prediction_length )
        # print('gru_out',output.shape)
        return output
    


# ==================================================
#  Homogeneous Graph+GRU
# ==================================================
class GCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):  # Corrected __init__
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(num_node_features, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, inputs, edge_index):  # Corrected argument name
        # inputs: (batch_size, num_stations, num_features)
        # edge_index: (2, num_edges)
        x = F.relu(self.gcn1(inputs, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        return x  # (batch_size, num_stations, hidden_dim)


class TGCN(nn.Module):
    def __init__(self, num_node_features,hidden_channels, hidden_units_gru,prediction_length):
        super(TGCN, self).__init__()
        self.gcn = GCN(num_node_features, hidden_channels)
        self.gru1 = nn.GRU(input_size=2*hidden_channels, hidden_size=hidden_units_gru, 
                          num_layers=1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(input_size=hidden_units_gru, hidden_size=hidden_units_gru, 
                          num_layers=1, batch_first=True, bidirectional=False)
        
        # --- Linear layers for projecting features to hidden_channels ---
        self.lin_flow = nn.Sequential(Linear(-1, hidden_channels),nn.ReLU())
        self.time_projection = nn.Sequential(Linear(-1, hidden_channels,nn.ReLU()))
        self.static_features_projection = nn.Sequential(Linear(-1, hidden_channels,nn.ReLU()))
        
        self.dense_output = nn.Linear(hidden_units_gru, prediction_length)



    def forward(self, x_dict):
      
        batch_size, seq_length,num_stations, _ = x_dict['features']['station_flows'].shape
        edge_index = x_dict['edge_indices'][('station','adjacent_to','station')][0,...]
     
        ### 只对poi和census features进行一次静态graph卷积，其它时间特征融入在GRU中
        static_features =torch.cat([x_dict['features']['station_poi'][0],x_dict['features']['census_zone'][0]],dim=-1)
        static_features=self.static_features_projection(static_features)
        gcn_output = self.gcn(static_features,edge_index )  # (num_stations, hidden_channels)
        gcn_output = gcn_output.unsqueeze(0).unsqueeze(0) # [1,1,num_stations, hidden_channels]
        gcn_output = gcn_output.expand(batch_size, seq_length, -1,-1) # [64, seq_length,num_stations, hidden_channels]
        # print(gcn_output.shape)
        
        ### 和其它时间特征连接
        time_features=torch.cat([x_dict['features']['station_flows'],x_dict['features']['hour_of_day'],x_dict['features']['day_of_week'],x_dict['features']['weather']],dim=-1) 
        time_features=self.time_projection(time_features)
        # print(time_features.shape)
        gcn_outputs_stacked=torch.concatenate([time_features,gcn_output],axis=-1)
        # Prepare for GRU: Reshape to treat each station independently
        gcn_outputs_stacked = gcn_outputs_stacked.permute(0, 2, 1, 3)  
        # print(gcn_outputs_stacked.shape)
        gru_input = gcn_outputs_stacked.reshape(batch_size * num_stations, seq_length, -1) 
        # Apply GRU layers
        gru1_output, _ = self.gru1(gru_input)  
        gru2_output, _ = self.gru2(gru1_output)  
        # print(gru2_output.shape) 
        gru2_output = gru2_output[:, -1, :]
        # Reshape and apply final dense layer
        output_reshaped = gru2_output.reshape(batch_size, num_stations, -1)  
        # print(output_reshaped.shape) # [64, 1807, 768]
        output = self.dense_output(output_reshaped)  # [64, num_stations, predict_sequence_length]

        return output


# ==================================================
# GRU
# ==================================================
class GRU(nn.Module):
    def __init__(self, hidden_channels, hidden_units_gru,prediction_length):
        super().__init__()
     
        self.time_projection = nn.Sequential(Linear(-1, hidden_channels,nn.ReLU()))
        self.static_features_projection = nn.Sequential(Linear(-1, hidden_channels,nn.ReLU()))
        
        self.gru1 = torch.nn.GRU(input_size=2*hidden_channels, hidden_size=hidden_units_gru, batch_first=True)
        self.gru2 = torch.nn.GRU(input_size=hidden_units_gru, hidden_size=hidden_units_gru, batch_first=True)
        self.dense_output = nn.Linear(hidden_units_gru, prediction_length)


        
    def forward(self, x_dict):
        batch_size, seq_length,num_stations, _ = x_dict['features']['station_flows'].shape
        static_features =torch.cat([x_dict['features']['station_poi'][0],x_dict['features']['census_zone'][0]],dim=-1)
        static_features=self.static_features_projection(static_features).unsqueeze(0).unsqueeze(0).expand(batch_size,seq_length,-1,-1)
        
        time_features=torch.cat([x_dict['features']['station_flows'],x_dict['features']['hour_of_day'],x_dict['features']['day_of_week'],x_dict['features']['weather']],dim=-1) 
        time_features=self.time_projection(time_features)
        
      
        # --- Temporal Processing with GRU ---
        # Reshape for GRU to treat each station's sequence independently - shape (batch_size * num_stations, sequence_length, out_channels)
        gru_input= torch.cat([time_features,static_features],dim=-1).permute(0, 2, 1,3)
        # print(gru_input.shape)
        gru_input = gru_input.reshape(gru_input.size(0) * gru_input.size(1), gru_input.size(2), gru_input.size(3)) 
        gru_out1, _ = self.gru1(gru_input) # gru_out: (batch_size * num_stations, sequence_length, hidden_units_gru)
        gru_out2, _ = self.gru2(gru_out1) 
        gru_out_last_timestep = gru_out2[:, -1, :]
        gru_out_reshaped = gru_out_last_timestep.reshape(batch_size,num_stations,-1) 
        output = self.dense_output(gru_out_reshaped)  # (batch_size, num_stations, prediction_length)
        # print('gru_out',output.shape)
        
        return output

# class TGCNModel(nn.Module):
#     def __init__(self, num_node_features,hidden_units_gcn, hidden_units_gru, predict_sequence_length, input_features):
#         super(TGCNModel, self).__init__()
#         self.gcn = SpatioTemporalGCN(num_node_features, hidden_units_gcn)
#         # self.lin = nn.Linear(hidden_units_gcn+16+16 , hidden_units_gru) # Correct input size
#         self.gru1 = nn.GRU(input_size=hidden_units_gru, hidden_size=hidden_units_gru, 
#                           num_layers=1, batch_first=True, bidirectional=False)
#         self.gru2 = nn.GRU(input_size=hidden_units_gru, hidden_size=hidden_units_gru, 
#                           num_layers=1, batch_first=True, bidirectional=False)
        
#         self.time_projection = nn.Linear(5, 16)
#         self.flows = nn.Linear(1, 16)
       
#         self.dense_output = nn.Linear(hidden_units_gru, predict_sequence_length)
#         self.input_features = input_features  # Store for initialization check


#     def forward(self, inputs, edge_index):
#         # inputs: (batch_size, sequence_length, num_stations, num_features), e.g., [64, 12, 500, 9]
#         # adjacency_matrix: (num_stations, num_stations), e.g., [500, 500]
#         batch_size, sequence_length, num_stations, num_features = inputs.shape
#         edge_index = edge_index[('station','adjacent_to','station')]
#         # print(edge_index.shape)
#         # 1. Apply Graph Convolution at each timestep
       
#         ## 取消注释如果需要对每一个时间步进行卷积
#         # gcn_outputs = []
#         # for t in range(sequence_length):
#         #     current_features = inputs[:, t, :, :]  # [64, 500, 9]
#         #     gcn_output = self.gcn(current_features,edge_index )  # [64, 500, hidden_units_gcn]
#         #     # print('gcn_output',gcn_output.shape) [64,500,16]
#         #     gcn_outputs.append(gcn_output)
#         # gcn_outputs.append(gcn_output)
#         # 2. Stack GCN outputs along time dimension
#         # gcn_outputs_stacked = torch.stack(gcn_outputs, dim=1)  # [64, 12, 500, hidden_units_gcn]
#         # print(gcn_outputs_stacked.shape) # [64, 12, 1807, 32]

#         ## 只对poi features进行一次静态卷积，其它时间特征融入在GRU中
#         current_features = inputs[:, 0, :, 1:9]  # [64, 500, 9]
#         gcn_output = self.gcn(current_features,edge_index )  # [64, 500, hidden_units_gcn]
#         # print('gcn_output',gcn_output.shape) [64,500,16]
#         gcn_output = gcn_output.expand(-1, num_stations, -1).unsqueeze(1) # [64, 1,1807, 16]
#         gcn_output = gcn_output.expand(-1, 12, -1,-1) # [64, 12,1807, 16]
#         ### 和其它时间特征连接
#         station_flow_features = inputs[...,0:1] 
#         weatherandother = inputs[...,9:]
#         # print(weatherandother.shape)
#         weatherandother,station_flow_features=self.time_projection(weatherandother),self.flows(station_flow_features)
#         gcn_outputs_stacked=torch.concatenate([station_flow_features,weatherandother,gcn_output],axis=-1)
       

#         # 3. Prepare for GRU: Reshape to treat each station independently
#         gcn_outputs_stacked = gcn_outputs_stacked.permute(0, 2, 1, 3)  # [64, 500, 12, hidden_units_gcn]
#         # print(gcn_outputs_stacked.shape)
#         gru_input = gcn_outputs_stacked.reshape(batch_size * num_stations, sequence_length, -1)  # [32000, 12, hidden_units_gcn]
   
#         # print('gru_input',gru_input.shape)
#         gru_input = self.lin(gru_input)  # [64, 1807, predict_sequence_length]

#         # 4. Apply GRU layers
#         gru1_output, _ = self.gru1(gru_input)  # [32000, 12, hidden_units_gru]
#         gru2_output, _ = self.gru2(gru1_output)  # [32000, hidden_units_gru] (last timestep only)
#         # print(gru2_output.shape) # [115648, 12, 64]
#         gru2_output = gru2_output[:, -1, :]

#         # 5. Reshape and apply final dense layer
#         output_reshaped = gru2_output.reshape(batch_size, num_stations, -1)  # [64, 1807, 64]
#         # print(output_reshaped.shape) # [64, 1807, 768]
#         output = self.dense_output(output_reshaped)  # [64, 1807, predict_sequence_length]

#         return output