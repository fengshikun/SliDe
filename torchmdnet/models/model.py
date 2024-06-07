import re
from typing import Optional, List, Tuple
import torch
from torch.autograd import grad
from torch import nn
from torch_scatter import scatter
from pytorch_lightning.utilities import rank_zero_warn
from torchmdnet.models import output_modules
from torchmdnet.models.wrappers import AtomFilter
from torchmdnet import priors
import warnings


def create_model(args, prior_model=None, mean=None, std=None):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "graph-network":
        from torchmdnet.models.torchmd_gn import TorchMD_GN

        is_equivariant = False
        representation_model = TorchMD_GN(
            num_filters=args["embedding_dimension"], aggr=args["aggr"], **shared_args
        )
    elif args["model"] == "transformer":
        from torchmdnet.models.torchmd_t import TorchMD_T

        is_equivariant = False
        representation_model = TorchMD_T(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformer":
        from torchmdnet.models.torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            md17=args["md17"],
            seperate_noise=args['seperate_noise'],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformern":
        from torchmdnet.models.torchmd_etn import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            md17=args["md17"],
            seperate_noise=args['seperate_noise'],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformerf2d":
        from torchmdnet.models.torchmd_etf2d import TorchMD_ETF2D

        is_equivariant = True
        representation_model = TorchMD_ETF2D(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            md17=args["md17"],
            seperate_noise=args['seperate_noise'],
            num_spherical=args['num_spherical'],
            **shared_args,
        )
    elif args["model"] == "equivariant-transformerf2d_org":
        from torchmdnet.models.torchmd_etf2d_org import TorchMD_ETF2D_ORG

        is_equivariant = True
        representation_model = TorchMD_ETF2D_ORG(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            md17=args["md17"],
            seperate_noise=args['seperate_noise'],
            num_spherical=args['num_spherical'],
            **shared_args,
        )

    elif args["model"] == "equiformer":
        is_equivariant = True
        pass
    elif args["model"] == "equiformer_oc20":
        is_equivariant = True
        pass
    elif args["model"] == "equiformer_v2":
        is_equivariant = True
        pass

    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # atom filter
    if not args["derivative"] and args["atom_filter"] > -1:
        representation_model = AtomFilter(representation_model, args["atom_filter"])
    elif args["atom_filter"] > -1:
        raise ValueError("Derivative and atom filter can't be used together")

    # prior model
    if args["prior_model"] and prior_model is None:
        assert "prior_args" in args, (
            f"Requested prior model {args['prior_model']} but the "
            f'arguments are lacking the key "prior_args".'
        )
        assert hasattr(priors, args["prior_model"]), (
            f'Unknown prior model {args["prior_model"]}. '
            f'Available models are {", ".join(priors.__all__)}'
        )
        # instantiate prior model if it was not passed to create_model (i.e. when loading a model)
        prior_model = getattr(priors, args["prior_model"])(**args["prior_args"])

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    if 'MolNetDataset' in args['dataset']:
        output_dim = 1
        if args['is_cls']:
            output_dim = args['cls_number']
            if output_dim == 1:
                output_dim = 2 # softmax for the single binary classification
            output_model = getattr(output_modules, output_prefix + args["output_model"])(
            args["embedding_dimension"], output_dim, args["activation"]
            )
        else: # regression
            output_dim = 1
            output_model = getattr(output_modules, output_prefix + args["output_model"])(
            args["embedding_dimension"], output_dim, args["activation"]
            )
    elif 'DockingDataset' in args['dataset']:
        output_dim = 5 # regression
        output_model = getattr(output_modules, output_prefix + args["output_model"])(
            args["embedding_dimension"], output_dim, args["activation"]
            )
    elif 'LBADataset' in args['dataset']:
        output_model = getattr(output_modules, args["output_model"])(
        args["embedding_dimension"], args["activation"]
        ) # normal MLP for LBADataset
    else:
        output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
        )

    # create the denoising output network
    output_model_noise = None
    if args['output_model_noise'] is not None:
        if args['bond_length_scale'] and args['model'] != 'equivariant-transformer':
            # output_bond_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            #     args["embedding_dimension"] * 2, args["activation"],
            # )
            # output_angle_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            #     args["embedding_dimension"] * 2, args["activation"],
            # )
            # output_dihedral_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            #     args["embedding_dimension"], args["activation"],
            # )


            # SIMPLE MLP Scalar head
            scalar_output_prefix = ''
            if args['use_edge_feat']:
                output_bond_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
                output_angle_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                    args["embedding_dimension"] * 2, args["activation"],
                )
                output_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                    args["embedding_dimension"] * 2, args["activation"],
                )
                output_rotate_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                    args["embedding_dimension"] * 2, args["activation"],
                )
            else:
                output_bond_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                    args["embedding_dimension"] * 2, args["activation"],
                )
                output_angle_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                    args["embedding_dimension"] * 3, args["activation"],
                )
                output_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                    args["embedding_dimension"] * 4, args["activation"],
                )
                output_rotate_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_noise"])(
                    args["embedding_dimension"] * 4, args["activation"],
                )


            # output model noise:
            output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"], args["activation"],
            ) # last one is coord denoise

            output_model_noise = nn.ModuleList([output_model_noise, output_bond_noise, output_angle_noise, output_dihedral_noise, output_rotate_dihedral_noise])

        elif args['dataset'] == 'PCQM4MV2_DihedralAll':
            # SIMPLE MLP Scalar head
            scalar_output_prefix = ''
            output_bond_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
            output_angle_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
            output_dihedral_noise = getattr(output_modules, scalar_output_prefix + args["output_model_ff_noise"])(
                args["embedding_dimension"] * 2, args["activation"],
            )
            output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"], args["activation"],
            )
            output_model_noise = nn.ModuleList([output_bond_noise, output_angle_noise, output_dihedral_noise, output_model_noise])
        else:
            output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"], args["activation"],
            )

    output_model_mask_atom = None
    if args['mask_atom']:
        output_model_mask_atom = getattr(output_modules, "MaskHead", args["embedding_dimension"])(args["embedding_dimension"], args["activation"],)

    if args['frad_guassian_scale'] != 0:
        output_model_ff = getattr(output_modules, output_prefix + args["output_model_noise"])(
                args["embedding_dimension"], args["activation"],
            )
    else:
        output_model_ff = None

    if args["model"] == "equiformer":
        from torchmdnet.models.equiformer import GraphAttentionTransformer
        is_equivariant = True
        irreps_in = '5x0e'
        atomref = None
        radius = 5
        num_basis = 128
        if args['dataset'] in {'QM9', 'oc20_is2re'}:
            output_dimension = '1x0e'
        else:
            output_dimension = '2x0e'
        model = GraphAttentionTransformer(
            irreps_in=irreps_in,
            irreps_node_embedding='128x0e+64x1e+32x2e', num_layers=6,
            irreps_node_attr='1x0e', irreps_sh='1x0e+1x1e+1x2e',
            output_dimension=output_dimension,
            max_radius=radius,
            number_of_basis=num_basis, fc_neurons=[64, 64],
            irreps_feature='512x0e',
            irreps_head='32x0e+16x1e+8x2e', num_heads=4, irreps_pre_attn=None,
            rescale_degree=False, nonlinear_message=True,
            irreps_mlp_mid='384x0e+192x1e+96x2e',
            norm_layer='layer',
            alpha_drop=0.2, proj_drop=0.0, out_drop=0.0, drop_path_rate=0.0,
            scale=None, atomref=atomref,
            # ---------------------------------------------
            prior_model=prior_model,
            reduce_op=args["reduce_op"],
            mean=mean,
            std=std,
            derivative=args["derivative"],
            output_model_noise=output_model_noise,
            output_model_mask_atom=output_model_mask_atom,
            position_noise_scale=args['position_noise_scale'],
            no_target_mean=args['no_target_mean'],
            seperate_noise=args['seperate_noise'],
            # bond length scale
            bond_length_scale=args['bond_length_scale'],
            dataset_name=args['dataset'], # may PCQM4MV2_DihedralAll
            use_edge_feat=args['use_edge_feat'], #
            derivative_var=args['derivative_var'],
            ff_learn_frad=(args['frad_guassian_scale'] != 0),
            output_model_ff = output_model_ff,
            )
        return model


    if args["model"] == "equiformer_oc20":
        from torchmdnet.models.equiformer import GraphAttentionTransformer
        is_equivariant = True
        irreps_in = '5x0e'
        atomref = None
        radius = 5
        num_basis = 128
        if args['dataset'] in {'QM9', 'oc20_is2re'}:
            output_dimension = '1x0e'
        else:
            output_dimension = '2x0e'
        model = GraphAttentionTransformer(
            irreps_node_embedding='256x0e+128x1e',
            num_layers=6,
            irreps_node_attr='1x0e',
            use_node_attr=False,
            irreps_sh='1x0e+1x1e',
            max_radius=5.0,
            number_of_basis=128,
            fc_neurons=[64, 64] ,
            use_atom_edge_attr=False,
            irreps_atom_edge_attr='1x0e',
            irreps_feature='512x0e',
            irreps_head='32x0e+16x1e',
            num_heads=8,
            irreps_pre_attn='256x0e+128x1e',
            rescale_degree=False,
            nonlinear_message=True,
            irreps_mlp_mid='768x0e+384x1e',
            norm_layer='layer',
            alpha_drop=0.2,
            proj_drop=0.0,
            out_drop=0.0,
            drop_path_rate=0.0,
            otf_graph=True,
            use_pbc=True,
            max_neighbors=500,
            # ---------------------------------------------
            prior_model=prior_model,
            reduce_op=args["reduce_op"],
            mean=mean,
            std=std,
            derivative=args["derivative"],
            output_model_noise=output_model_noise,
            output_model_mask_atom=output_model_mask_atom,
            position_noise_scale=args['position_noise_scale'],
            no_target_mean=args['no_target_mean'],
            seperate_noise=args['seperate_noise'],
            # bond length scale
            bond_length_scale=args['bond_length_scale'],
            dataset_name=args['dataset'], # may PCQM4MV2_DihedralAll
            use_edge_feat=args['use_edge_feat'], #
            derivative_var=args['derivative_var'],
            ff_learn_frad=(args['frad_guassian_scale'] != 0),
            output_model_ff = output_model_ff,
            )
        return model

    elif args["model"] == "equiformer_v2":
        from torchmdnet.models.equiformer_v2 import EquiformerV2_OC20
        is_equivariant = True
        model = EquiformerV2_OC20(
            num_atoms = None,
            bond_feat_dim = None,
            num_targets = None,
            use_pbc=True,
            regress_forces=False,
            otf_graph=True,
            max_neighbors=50,
            max_radius=5.0,
            max_num_elements=118,
            num_layers=6,
            sphere_channels=128,
            attn_hidden_channels=128,
            num_heads=8,
            # ---------------------------------------------
            prior_model=prior_model,
            reduce_op=args["reduce_op"],
            mean=mean,
            std=std,
            derivative=args["derivative"],
            output_model_noise=output_model_noise,
            output_model_mask_atom=output_model_mask_atom,
            position_noise_scale=args['position_noise_scale'],
            no_target_mean=args['no_target_mean'],
            seperate_noise=args['seperate_noise'],
            # bond length scale
            bond_length_scale=args['bond_length_scale'],
            dataset_name=args['dataset'], # may PCQM4MV2_DihedralAll
            use_edge_feat=args['use_edge_feat'], #
            derivative_var=args['derivative_var'],
            ff_learn_frad=(args['frad_guassian_scale'] != 0),
            output_model_ff = output_model_ff,
            )
        return model

    # combine representation and output network
    model = TorchMD_Net(
        representation_model,
        output_model,
        prior_model=prior_model,
        reduce_op=args["reduce_op"],
        mean=mean,
        std=std,
        derivative=args["derivative"],
        output_model_noise=output_model_noise,
        output_model_mask_atom=output_model_mask_atom,
        position_noise_scale=args['position_noise_scale'],
        no_target_mean=args['no_target_mean'],
        seperate_noise=args['seperate_noise'],
        # bond length scale
        bond_length_scale=args['bond_length_scale'],
        dataset_name=args['dataset'], # may PCQM4MV2_DihedralAll
        use_edge_feat=args['use_edge_feat'], #
        derivative_var=args['derivative_var'],
        ff_learn_frad=(args['frad_guassian_scale'] != 0),
        output_model_ff = output_model_ff
    )
    return model


def load_model(filepath, args=None, device="cpu", mean=None, std=None, **kwargs):
    ckpt = torch.load(filepath, map_location="cpu")
    if args is None:
        args = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if not key in args:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        args[key] = value

    model = create_model(args)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}

    # NOTE for debug
    new_state_dict = {}
    for k, v in state_dict.items():
        # if 'pos_normalizer' not in k:
        if 'output_model_noise.0' in k:
            k = k.replace('output_model_noise.0', 'output_model_noise')
        if 'head.2' in k:
            continue
        new_state_dict[k] = v
    
    
    current_model_dict = model.state_dict()
    # ommit mismatching shape
    new_state_dict2 = {}
    for k in current_model_dict:
        if k in new_state_dict:
            if current_model_dict[k].size() == new_state_dict[k].size():
                new_state_dict2[k] = new_state_dict[k]
            else:
                print(f'warning {k} shape mismatching, not loaded')
                new_state_dict2[k] = current_model_dict[k]
    
    # new_state_dict2 = {k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), new_state_dict.values())}
    # for k,v in zip(current_model_dict.keys(), new_state_dict.values()):
    #     if v.size()!=current_model_dict[k].size():
    #         print(f'warning {k} shape mismatching, not loaded')
    
    # loading_return = model.load_state_dict(state_dict, strict=False)
    loading_return = model.load_state_dict(new_state_dict2, strict=False)
    
    # loading_return = model.load_state_dict(state_dict, strict=False)
    # loading_return = model.load_state_dict(new_state_dict, strict=False)
    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        # assert all(("output_model_noise" in k or "pos_normalizer" in k) for k in loading_return.unexpected_keys)
        pass
    # assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"
    if len(loading_return.missing_keys) > 0:
        print(f'warning:  load model missing keys {loading_return.missing_keys}')

    if isinstance(mean, torch.Tensor): # for docking dataset test
        model.mean = mean
        model.std = std
        return model.to(device)


    if mean:
        if isinstance(mean, list):
            model.mean = mean[0]
            model.register_buffer("dmean", mean[1])
        else:
            model.mean = mean
    if std:
        if isinstance(std, list):
            model.std = std[0]
            model.register_buffer("dstd", std[1])
        else:
            model.std = std

    return model.to(device)


class TorchMD_Net(nn.Module):
    def __init__(
        self,
        representation_model,
        output_model,
        prior_model=None,
        reduce_op="add",
        mean=None,
        std=None,
        derivative=False,
        output_model_noise=None,
        position_noise_scale=0.,
        no_target_mean=False,
        seperate_noise=False,
        output_model_mask_atom=None,
        bond_length_scale=0.,
        derivative_var=0.,
        dataset_name=None,
        use_edge_feat=False,
        ff_learn_frad=False,
        output_model_ff=None,
    ):
        super(TorchMD_Net, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.prior_model = prior_model
        if not output_model.allow_prior_model and prior_model is not None:
            self.prior_model = None
            rank_zero_warn(
                (
                    "Prior model was given but the output model does "
                    "not allow prior models. Dropping the prior model."
                )
            )

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale
        self.no_target_mean = no_target_mean
        self.seperate_noise = seperate_noise

        self.bond_length_scale = bond_length_scale

        self.output_model_mask_atom = output_model_mask_atom

        self.dataset_name = dataset_name
        self.use_edge_feat = use_edge_feat

        self.ff_learn_frad = ff_learn_frad # use frad when regression on the force field

        self.output_model_ff = output_model_ff

        mean = torch.scalar_tensor(0) if mean is None else mean
        if isinstance(mean, list):
            self.register_buffer("mean", mean[0]) # md17 y
            self.register_buffer("dmean", mean[1]) # md17 dy
        else:
            self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1) if std is None else std
        if isinstance(std, list):
            self.register_buffer("std", std[0]) # md17 y
            self.register_buffer("dstd", std[1]) # md17 dy
        else:
            self.register_buffer("std", std)

        if self.position_noise_scale > 0 and not self.no_target_mean:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
        else:
            self.pos_normalizer = None

        if self.bond_length_scale > 0 and not self.no_target_mean:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))
            if self.ff_learn_frad:
                self.ff_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

            self.bond_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.angle_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.dihedral_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.rotate_dihedral_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            # TODO: self.output_model_noise: List
            hidden_channels = self.representation_model.hidden_channels

            self.angle_ijk_proj = nn.Linear(hidden_channels * 3, hidden_channels * 2)
            self.dihedral_jk_proj = nn.Linear(hidden_channels * 2, hidden_channels)

        elif self.dataset_name == 'PCQM4MV2_DihedralAll':
            self.bond_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.angle_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
            self.dihedral_pos_normalizer = AccumulatedNormalization(accumulator_shape=(1,))
        else:
            self.bond_pos_normalizer = None
            self.angle_pos_normalizer = None
            self.dihedral_pos_normalizer = None

        self.derivative_var = derivative_var
        if self.derivative_var:
            self.reg_dev_normalizer = AccumulatedNormalization(accumulator_shape=(1,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()


    def extract_edge_feat(self, batch_org, e, edge_index):
        bond_idx = batch_org.bond_target[:, :2].to(torch.long)
        edge_feat = e[0]
        edge_idx = edge_index.T
        edge_idx_s = edge_idx.unsqueeze(0)
        # bond_idx_s = torch.zeros((2 * bond_idx.size(0), bond_idx.size(1)), dtype=bond_idx.dtype, device=bond_idx.device)
        # bond_idx_s[0::2] = bond_idx # orginal bond_idx
        # bond_idx_s[1::2] = bond_idx.flip(1)
        # bond_idx_s = bond_idx_s.unsqueeze(1)

        bond_idx_s = bond_idx.unsqueeze(1)
        bond_idx_r_s = bond_idx.flip(1).unsqueeze(1)

        mask = torch.all(edge_idx_s == bond_idx_s, dim=-1)
        indices = torch.nonzero(mask, as_tuple=False)


        mask_r = torch.all(edge_idx_s == bond_idx_r_s, dim=-1)
        indices_r = torch.nonzero(mask_r, as_tuple=False)

        bond_valid_idx = indices[:, 0]
        bond_r_valid_idx = indices_r[:, 0]
        bond_valid = indices[:, 1][torch.isin(bond_valid_idx, bond_r_valid_idx)]
        bond_r_valid = indices_r[:, 1][torch.isin(bond_r_valid_idx, bond_valid_idx)]

        bond_valid_idx = bond_valid_idx[torch.isin(bond_valid_idx, bond_r_valid_idx)]

        bond_feat = edge_feat[bond_valid]
        bond_r_feat = edge_feat[bond_r_valid]

        # bond_valid_idx = indices[0::2][:,0] // 2

        # bond_valid_feat = edge_feat[indices[:, 1]]
        bond_valid_feat = torch.concat([bond_feat, bond_r_feat], dim=1)

        # angle
        angle_idx = batch_org.angle_target[:, :3].to(torch.long) # get i, j, k
        # use (j, i) and (j, k)
        bond_ji= angle_idx[:,:2].flip(1)
        bond_jk = angle_idx[:,1:]
        bond_ji_s = bond_ji.unsqueeze(1)
        mask_ji = torch.all(edge_idx_s == bond_ji_s, dim=-1)
        ji_indices = torch.nonzero(mask_ji, as_tuple=False)
        bond_jk_s = bond_jk.unsqueeze(1)
        mask_jk = torch.all(edge_idx_s == bond_jk_s, dim=-1)
        jk_indices = torch.nonzero(mask_jk, as_tuple=False)

        ji_bond_valid_idx = ji_indices[:, 0]
        jk_bond_valid_idx = jk_indices[:, 0]
        ji_bond_valid = ji_indices[:, 1][torch.isin(ji_bond_valid_idx, jk_bond_valid_idx)]
        jk_bond_valid = jk_indices[:, 1][torch.isin(jk_bond_valid_idx, ji_bond_valid_idx)]

        angle_valid_idx = ji_bond_valid_idx[torch.isin(ji_bond_valid_idx, jk_bond_valid_idx)]


        # dihedral
        ji_bond_feat = edge_feat[ji_bond_valid]
        jk_bond_feat = edge_feat[jk_bond_valid]
        angle_valid_feat = torch.concat([ji_bond_feat, jk_bond_feat], dim=1)
        # angle_noise_pred = self.output_model_noise[1].pre_reduce(angle_valid_feat, v, z, pos, batch)
        # NOTE ommit dihedral target
        # dihedral_idx = batch_org.dihedral_target[:, :4].to(torch.long)[:,1:3]# only pick j,k

        # dihedral_idx_s = dihedral_idx.unsqueeze(1)
        # dihedral_idx_r_s = dihedral_idx.flip(1).unsqueeze(1)

        # mask = torch.all(edge_idx_s == dihedral_idx_s, dim=-1)
        # indices = torch.nonzero(mask, as_tuple=False)


        # mask_r = torch.all(edge_idx_s == dihedral_idx_r_s, dim=-1)
        # indices_r = torch.nonzero(mask_r, as_tuple=False)

        # dihedral_valid_idx = indices[:, 0]
        # dihedral_r_valid_idx = indices_r[:, 0]
        # dihedral_valid = indices[:, 1][torch.isin(dihedral_valid_idx, dihedral_r_valid_idx)]
        # dihedral_r_valid = indices_r[:, 1][torch.isin(dihedral_r_valid_idx, dihedral_valid_idx)]

        # dihedral_valid_idx = dihedral_valid_idx[torch.isin(dihedral_valid_idx, dihedral_r_valid_idx)]

        # dihedral_feat = edge_feat[dihedral_valid]
        # dihedral_r_feat = edge_feat[dihedral_r_valid]

        # # bond_valid_idx = indices[0::2][:,0] // 2

        # # bond_valid_feat = edge_feat[indices[:, 1]]
        # dihedral_valid_feat = torch.concat([dihedral_feat, dihedral_r_feat], dim=1)

        # if 'rotate_dihedral_target' in batch_org:
        #     rotate_dihedral_idx = batch_org.rotate_dihedral_target[:, :4].to(torch.long)[:,1:3]# only pick j,k

        #     rotate_dihedral_idx_s = rotate_dihedral_idx.unsqueeze(1)
        #     rotate_dihedral_idx_r_s = rotate_dihedral_idx.flip(1).unsqueeze(1)

        #     mask = torch.all(edge_idx_s == rotate_dihedral_idx_s, dim=-1)
        #     indices = torch.nonzero(mask, as_tuple=False)


        #     mask_r = torch.all(edge_idx_s == rotate_dihedral_idx_r_s, dim=-1)
        #     indices_r = torch.nonzero(mask_r, as_tuple=False)

        #     rotate_dihedral_valid_idx = indices[:, 0]
        #     rotate_dihedral_r_valid_idx = indices_r[:, 0]
        #     rotate_dihedral_valid = indices[:, 1][torch.isin(rotate_dihedral_valid_idx, rotate_dihedral_r_valid_idx)]
        #     rotate_dihedral_r_valid = indices_r[:, 1][torch.isin(rotate_dihedral_r_valid_idx, rotate_dihedral_valid_idx)]

        #     rotate_dihedral_valid_idx = rotate_dihedral_valid_idx[torch.isin(rotate_dihedral_valid_idx, rotate_dihedral_r_valid_idx)]

        #     rotate_dihedral_feat = edge_feat[rotate_dihedral_valid]
        #     rotate_dihedral_r_feat = edge_feat[rotate_dihedral_r_valid]

        #     # bond_valid_idx = indices[0::2][:,0] // 2

        #     # bond_valid_feat = edge_feat[indices[:, 1]]
        #     rotate_dihedral_valid_feat = torch.concat([rotate_dihedral_feat, rotate_dihedral_r_feat], dim=1)
        #     return bond_valid_idx, bond_valid_feat, angle_valid_idx, angle_valid_feat, dihedral_valid_idx, dihedral_valid_feat, rotate_dihedral_valid_idx, rotate_dihedral_valid_feat
        # return bond_valid_idx, bond_valid_feat, angle_valid_idx, angle_valid_feat, dihedral_valid_idx, dihedral_valid_feat
        return bond_valid_idx, bond_valid_feat, angle_valid_idx, angle_valid_feat



        # dihedral_idx_s = torch.zeros((2 * dihedral_idx.size(0), dihedral_idx.size(1)), dtype=dihedral_idx.dtype, device=dihedral_idx.device)
        # dihedral_idx_s[0::2] = dihedral_idx # orginal bond_idx
        # dihedral_idx_s[1::2] = dihedral_idx.flip(1)
        # dihedral_idx_s = dihedral_idx_s.unsqueeze(1)
        # mask = torch.all(edge_idx_s == dihedral_idx_s, dim=-1)
        # indices = torch.nonzero(mask, as_tuple=False)

        # dihedral_valid_idx = indices[0::2][:,0] // 2

        # dihedral_valid_feat = edge_feat[indices[:, 1]]
        # dihedral_valid_feat = torch.concat([dihedral_valid_feat[0::2], dihedral_valid_feat[1::2]], dim=1)
        # dihedral_noise_pred = self.output_model_noise[2].pre_reduce(dihedral_valid_feat, v, z, pos, batch)

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None, batch_org = None, frad_pos=False):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        if self.derivative:
            pos.requires_grad_(True)

        if self.seperate_noise:
            x, v, nv, z, pos, batch = self.representation_model(z, pos, batch=batch)
        else:
            # run the potentially wrapped representation model
            if self.dataset_name == 'PCQM4MV2_DihedralAll' or self.use_edge_feat:
                x, v, z, pos, batch, e, edge_index = self.representation_model(z, pos, batch=batch, return_e=True)
            elif batch_org is not None and 'type_mask' in batch_org and self.representation_model.max_z > 200:
                x, v, z, pos, batch = self.representation_model(z, pos, batch=batch, type_idx=batch_org.type_mask)
            else:
                x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)
            nv = None


        # whether mask or not
        mask_logits = None
        if self.output_model_mask_atom is not None:
            mask_logits = self.output_model_mask_atom.pre_reduce(x)



        if self.bond_length_scale > 0: # and not self.representation_model.md17:
            if self.use_edge_feat:
                # bond_valid_idx, bond_valid_feat, angle_valid_idx, angle_valid_feat, dihedral_valid_idx, dihedral_valid_feat, rotate_dihedral_valid_idx, rotate_dihedral_valid_feat = self.extract_edge_feat(batch_org, e, edge_index)
                bond_valid_idx, bond_valid_feat, angle_valid_idx, angle_valid_feat = self.extract_edge_feat(batch_org, e, edge_index)

            else:
            # collect bond featrue
                bond_idx = batch_org.bond_target[:, :2].to(torch.long)
                bond_i_x = x[bond_idx[:, 0]]
                bond_j_x = x[bond_idx[:, 1]]
                bond_i_v = v[bond_idx[:, 0]]
                bond_j_v = v[bond_idx[:, 1]]

                # concat i and j
                bond_x = torch.cat([bond_i_x, bond_j_x], axis=1) # X * 512
                bond_v = torch.cat([bond_i_v, bond_j_v], axis=2) # X * 512

                # collect angle featrue
                angle_idx = batch_org.angle_target[:, :3].to(torch.long)
                angle_i_x = x[angle_idx[:, 0]]
                angle_j_x = x[angle_idx[:, 1]]
                angle_k_x = x[angle_idx[:, 2]]
                # angle_x = self.angle_ijk_proj(torch.cat([angle_i_x, angle_j_x, angle_k_x], axis=1))
                angle_x = torch.cat([angle_i_x, angle_j_x, angle_k_x], axis=1)

                angle_i_v = v[angle_idx[:, 0]]
                angle_j_v = v[angle_idx[:, 1]]
                angle_k_v = v[angle_idx[:, 2]]

                angle_ji_v = angle_i_v - angle_j_v # TODO direction?
                angle_jk_v = angle_k_v - angle_j_v # TODO direction?
                angle_v = torch.cat([angle_ji_v, angle_jk_v], axis=2)

                # collect dihedral featrue
                dihedral_idx = batch_org.dihedral_target[:, :4].to(torch.long)
                # only pick j,k
                dihedral_i_x = x[dihedral_idx[:, 0]]
                dihedral_j_x = x[dihedral_idx[:, 1]]
                dihedral_k_x = x[dihedral_idx[:, 2]]
                dihedral_l_x = x[dihedral_idx[:, 3]]
                # dihedral_x = self.dihedral_jk_proj(torch.cat([dihedral_j_x, dihedral_k_x], axis=1))
                dihedral_x = torch.cat([dihedral_i_x, dihedral_j_x, dihedral_k_x, dihedral_l_x], axis=1)


                dihedral_j_v = v[dihedral_idx[:, 0]]
                dihedral_k_v = v[dihedral_idx[:, 1]]
                dihedral_v = dihedral_k_v - dihedral_j_v # TODO direction?


                rotate_dihedral_idx = batch_org.rotate_dihedral_target[:, :4].to(torch.long)
                # only pick j,k
                rotate_dihedral_i_x = x[rotate_dihedral_idx[:, 0]]
                rotate_dihedral_j_x = x[rotate_dihedral_idx[:, 1]]
                rotate_dihedral_k_x = x[rotate_dihedral_idx[:, 2]]
                rotate_dihedral_l_x = x[rotate_dihedral_idx[:, 3]]
                # dihedral_x = self.dihedral_jk_proj(torch.cat([dihedral_j_x, dihedral_k_x], axis=1))
                rotate_dihedral_x = torch.cat([rotate_dihedral_i_x, rotate_dihedral_j_x, rotate_dihedral_k_x, rotate_dihedral_l_x], axis=1)


                rotate_dihedral_j_v = v[rotate_dihedral_idx[:, 0]]
                rotate_dihedral_k_v = v[rotate_dihedral_idx[:, 1]]
                rotate_dihedral_v = rotate_dihedral_k_v - rotate_dihedral_j_v # TODO direction?

        elif self.dataset_name == 'PCQM4MV2_DihedralAll':
            bond_valid_idx, bond_valid_feat, angle_valid_idx, angle_valid_feat, dihedral_valid_idx, dihedral_valid_feat = self.extract_edge_feat(batch_org, e, edge_index)

        # predict noise
        noise_pred = None
        if self.output_model_noise is not None:
            if nv is not None:
                noise_pred = self.output_model_noise.pre_reduce(x, nv, z, pos, batch)
            else:
                if self.bond_length_scale > 0: # and not self.representation_model.md17
                    if self.use_edge_feat:
                        bond_noise_pred = self.output_model_noise[1].pre_reduce(bond_valid_feat, v, z, pos, batch).mean(axis=1)
                        angle_noise_pred = self.output_model_noise[2].pre_reduce(angle_valid_feat, v, z, pos, batch).mean(axis=1)
                        # dihedral_noise_pred = self.output_model_noise[2].pre_reduce(dihedral_valid_feat, v, z, pos, batch).mean(axis=1)
                        # rotate_dihedral_noise_pred = self.output_model_noise[3].pre_reduce(rotate_dihedral_valid_feat, v, z, pos, batch).mean(axis=1)
                        if 'pos_frad' in batch_org.keys and not frad_pos: # multi task, ff learn and frad, use another head for fflearn
                            noise_pred = self.output_model_ff.pre_reduce(x, v, z, pos, batch)
                        else:
                            noise_pred = self.output_model_noise[0].pre_reduce(x, v, z, pos, batch)

                        noise_pred = [noise_pred, bond_noise_pred, angle_noise_pred, bond_valid_idx, angle_valid_idx]
                        # noise_pred = [bond_noise_pred, angle_noise_pred, dihedral_noise_pred, rotate_dihedral_noise_pred, bond_valid_idx, angle_valid_idx, dihedral_valid_idx, rotate_dihedral_valid_idx]
                    else:
                        bond_noise_pred = self.output_model_noise[0].pre_reduce(bond_x, bond_v, z, pos, batch).mean(axis=1)
                        angle_noise_pred = self.output_model_noise[1].pre_reduce(angle_x, angle_v, z, pos, batch).mean(axis=1)
                        dihedral_noise_pred = self.output_model_noise[2].pre_reduce(dihedral_x, dihedral_v, z, pos, batch).mean(axis=1)
                        rotate_dihedral_noise_pred = self.output_model_noise[3].pre_reduce(rotate_dihedral_x, rotate_dihedral_v, z, pos, batch).mean(axis=1)

                        noise_pred = [bond_noise_pred, angle_noise_pred, dihedral_noise_pred, rotate_dihedral_noise_pred]
                elif self.dataset_name == 'PCQM4MV2_DihedralAll':
                    bond_noise_pred = self.output_model_noise[0].pre_reduce(bond_valid_feat, v, z, pos, batch) # no use of v, z, pos, batch
                    angle_noise_pred = self.output_model_noise[1].pre_reduce(angle_valid_feat, v, z, pos, batch)
                    dihedral_noise_pred = self.output_model_noise[2].pre_reduce(dihedral_valid_feat, v, z, pos, batch)


                    noise_pred = self.output_model_noise[3].pre_reduce(x, v, z, pos, batch)

                    noise_pred = [noise_pred, bond_noise_pred, angle_noise_pred, dihedral_noise_pred, bond_valid_idx, angle_valid_idx, dihedral_valid_idx]
                else:
                    if batch_org is not None and 'pos_frad' in batch_org.keys and not frad_pos: # multi task, ff learn and frad, use another head for fflearn
                        noise_pred = self.output_model_ff.pre_reduce(x, v, z, pos, batch)
                    else:
                        noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)
                    # noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)

        # apply the output network
        if 'EquivariantScalarMolInject' in str(type(self.output_model)):
            x = self.output_model.pre_reduce(x, v, z, pos, batch, batch_org.smi)
        else:
            x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # scale by data standard deviation
        if self.std is not None:
            x = x * self.std

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        if x.shape[0] == batch.shape[0]: # in case have scattered
            out = scatter(x, batch, dim=0, reduce=self.reduce_op)
        else:
            out = x

        # shift by data mean
        if self.mean is not None:
            out = out + self.mean

        # apply output model after reduction
        out = self.output_model.post_reduce(out)


        # if noise_pred is not None and hasattr(self, 'dmean'):
        #     noise_pred = noise_pred * self.dstd + self.dmean

        # compute gradients with respect to coordinates
        if self.derivative:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
            dy = grad(
                [out],
                [pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            if dy is None:
                raise RuntimeError("Autograd returned None for the force prediction.")
            return out, noise_pred, -dy
        # TODO: return only `out` once Union typing works with TorchScript (https://github.com/pytorch/pytorch/pull/53180)
        # return out, noise_pred, None
        return out, noise_pred, mask_logits


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""
    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return ((batch - self.mean) / self.std)
