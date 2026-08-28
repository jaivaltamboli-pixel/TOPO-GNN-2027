import numpy as np
import torch
import stim

def extract_complete_dem_graph(dem, num_dets, det_coords, d):
    """
    Parses Stim DEM instructions to extract:
    1. Detector-to-Detector edges (two-detector mechanisms)
    2. Detector-to-Boundary edges (single-detector mechanisms)
    3. Exact physical error probabilities and canonical LLR weights w_0
    4. Anisotropy flags derived from physical space-time orientations
    5. Direct DEM fault-mechanism-index-to-edge mapping table
    """
    bnd_z_idx = num_dets
    bnd_x_idx = num_dets + 1
    edge_dict = {}
    dem_fault_to_edge = []

    fault_idx = 0
    for instruction in dem:
        if instruction.type == 'error':
            p_val = instruction.args_copy()[0]
            targets = instruction.targets_copy()
            dets = [t.val for t in targets if t.is_relative_detector_id()]
            obs = [t.val for t in targets if t.is_logical_observable_id()]
            has_obs = len(obs) > 0

            llr = float(-np.log(max(p_val, 1e-12) / (1.0 - min(p_val, 0.4999))))

            edge_key = None
            if len(dets) == 2:
                u, v = min(dets[0], dets[1]), max(dets[0], dets[1])
                if u < num_dets and v < num_dets:
                    cu, cv = det_coords.get(u, [0, 0, 0]), det_coords.get(v, [0, 0, 0])
                    dx, dy, dt = abs(cu[0] - cv[0]), abs(cu[1] - cv[1]), abs(cu[2] - cv[2])
                    is_dephase = (dt == 0 and dy >= dx)
                    edge_key = (u, v)
                    
                    if edge_key not in edge_dict or p_val > edge_dict[edge_key]['p']:
                        edge_dict[edge_key] = {
                            'p': p_val, 'weight': llr, 'is_dephase': is_dephase,
                            'is_bnd_z': False, 'is_bnd_x': False, 'has_obs': has_obs
                        }
            elif len(dets) == 1:
                u = dets[0]
                if u < num_dets:
                    cu = det_coords.get(u, [0, 0, 0])
                    is_z_bnd = (cu[0] <= 1.0 or cu[0] >= d - 1.0)
                    v = bnd_z_idx if is_z_bnd else bnd_x_idx
                    edge_key = (u, v)
                    
                    if edge_key not in edge_dict or p_val > edge_dict[edge_key]['p']:
                        edge_dict[edge_key] = {
                            'p': p_val, 'weight': llr, 'is_dephase': False,
                            'is_bnd_z': is_z_bnd, 'is_bnd_x': not is_z_bnd, 'has_obs': has_obs
                        }
            
            dem_fault_to_edge.append(edge_key)
            fault_idx += 1

    return edge_dict, bnd_z_idx, bnd_x_idx, dem_fault_to_edge

def extract_active_subgraph_tensors(s_vec, det_coords, edge_dict, bnd_z_idx, bnd_x_idx, d, device, active_fault_pairs=None):
    active_dets = np.where(s_vec)[0].tolist()
    subgraph_nodes = list(active_dets)
    
    if bnd_z_idx not in subgraph_nodes:
        subgraph_nodes.append(bnd_z_idx)
    if bnd_x_idx not in subgraph_nodes:
        subgraph_nodes.append(bnd_x_idx)
        
    node_to_local = {node_id: idx for idx, node_id in enumerate(subgraph_nodes)}
    num_sub_nodes = len(subgraph_nodes)
    
    node_mat_6d = np.zeros((num_sub_nodes, 6), dtype=np.float32)
    node_mat_4d = np.zeros((num_sub_nodes, 4), dtype=np.float32)
    
    for local_idx, global_id in enumerate(subgraph_nodes):
        if global_id == bnd_z_idx:
            node_mat_6d[local_idx] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            node_mat_4d[local_idx] = [0.0, 0.0, 0.0, 0.0]
        elif global_id == bnd_x_idx:
            node_mat_6d[local_idx] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            node_mat_4d[local_idx] = [0.0, 0.0, 0.0, 0.0]
        else:
            c = det_coords.get(global_id, [0.0, 0.0, 0.0])
            dist_rough = float(min(c[0], d - c[0]))
            dist_smooth = float(min(c[1], d - c[1]))
            node_mat_6d[local_idx] = [float(s_vec[global_id]), c[0], c[1], c[2], dist_rough, dist_smooth]
            node_mat_4d[local_idx] = [float(s_vec[global_id]), c[0], c[1], c[2]]

    src_list, dst_list, attr_list, is_par_list, edge_targets, global_pairs = [], [], [], [], [], []
    
    for (u, v), props in edge_dict.items():
        if u in node_to_local and v in node_to_local:
            lu, lv = node_to_local[u], node_to_local[v]
            src_list.extend([lu, lv])
            dst_list.extend([lv, lu])
            global_pairs.extend([(u, v), (v, u)])
            
            w = props['weight']
            is_bz = 1.0 if props['is_bnd_z'] else 0.0
            is_bx = 1.0 if props['is_bnd_x'] else 0.0
            
            attr_list.extend([[w, props['p'], is_bz, is_bx], [w, props['p'], is_bz, is_bx]])
            is_par_list.extend([props['is_dephase'], props['is_dephase']])
            
            # Ground-truth physical edge supervision
            is_active_edge = 1.0 if (active_fault_pairs is not None and ((u, v) in active_fault_pairs or (v, u) in active_fault_pairs)) else 0.0
            edge_targets.extend([[is_active_edge], [is_active_edge]])

    x4 = torch.tensor(node_mat_4d, dtype=torch.float32, device=device)
    x6 = torch.tensor(node_mat_6d, dtype=torch.float32, device=device)
    
    if len(src_list) > 0:
        e_idx = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        e_attr = torch.tensor(attr_list, dtype=torch.float32, device=device)
        e_par = torch.tensor(is_par_list, dtype=torch.bool, device=device)
        e_targ = torch.tensor(edge_targets, dtype=torch.float32, device=device)
    else:
        e_idx = torch.zeros((2, 0), dtype=torch.long, device=device)
        e_attr = torch.zeros((0, 4), dtype=torch.float32, device=device)
        e_par = torch.zeros((0,), dtype=torch.bool, device=device)
        e_targ = torch.zeros((0, 1), dtype=torch.float32, device=device)
        
    s_t = torch.tensor(node_mat_4d[:, :1], dtype=torch.float32, device=device)
    return x4, x6, e_idx, e_attr, e_par, s_t, e_targ, global_pairs
