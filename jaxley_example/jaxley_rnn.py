import jaxley as jx
from jaxley.channels import Leak, HH
from jaxley.connect import fully_connect, connect
from jaxley.synapses import IonotropicSynapse

import numpy as np
import jax.numpy as jnp

def build_rnn(RNN_params,verbose=True):

    # Define a compartment, branch, and cell
    comp = jx.Compartment()
    branch = jx.Branch(comp,ncomp=1)
    cell = jx.Cell(branch, parents=[-1])

    num_cells=RNN_params["n_rec"]  + RNN_params["n_out"]
    n_rec = RNN_params["n_rec"]
    n_inh =  RNN_params["n_inh"]
    n_out = RNN_params["n_out"]
    init_gain=RNN_params["init_gain"]
    leak_out = True

    net = jx.Network([cell for _ in range(num_cells)])
    rec = net.cell(range(num_cells-n_out))
    readout = net.cell([range(num_cells-n_out, num_cells)])
    negative_inds = np.arange(n_inh).tolist()
    rec_conn_prob=.2
    # Connect units randomly
    for i in range(num_cells-n_out):
        for j in range(num_cells-n_out):
            if np.random.binomial(1, rec_conn_prob):
                connect(
                    net.cell(i).branch(0).comp(0),
                    net.cell(j).branch(0).comp(0),
                    IonotropicSynapse(),
                )
    # Connect the readout units
    fully_connect(rec, readout, IonotropicSynapse())
            

    # Insert mechanisms
    for i in range(num_cells - n_out):
        net.cell(i).insert(HH())
    for i in range(n_out):
        if leak_out:
            net.cell(num_cells - n_out + i).insert(Leak())
            net.cell(num_cells - n_out + i).set("Leak_gLeak", 3e-2)
            if verbose:
                print("Leak out")
        else:
            net.cell(num_cells - n_out + i).insert(HH())
            if verbose:
                print("HH out")



    negative_inds = np.arange(n_inh).tolist()
    conn_matrix = init_inh_ex_gS(net, negative_inds, init_gain, out_indices=[i for i in range(num_cells - n_out, num_cells)],
                                  return_matrix=True, verbose=verbose)

    # Set some parameters
    net.set("IonotropicSynapse_k_minus", RNN_params["k_minus"]) 
    net.set("v", -67.0)
    net.init_states()

    # Initialize input weights
    n_inp = RNN_params['n_stim'] + 1 #stimuli and fixation
    input_weights = abs(np.random.uniform(0, 1, size = (n_rec,n_inp))) * RNN_params['inp_scale']
    in_conn_prob= RNN_params['in_conn_prob']
    if in_conn_prob < 1.0:
        input_weights_mask = np.zeros((n_rec,n_inp))
        for i in range(n_inp):
            n_nonzero = int(n_rec*in_conn_prob)
            nonzero_indices = np.random.choice(np.arange(n_rec), size=n_nonzero, replace=False)
            input_weights_mask[nonzero_indices, i] = 1
    else:
        input_weights_mask = np.ones((n_rec,n_inp))
    input_weights = [{"input_weights": jnp.asarray(input_weights)}]

    return net, conn_matrix, input_weights,input_weights_mask


def init_inh_ex_gS(
    network, inh_indices, gain, out_indices = [],e_syn_inh=-75, e_syn_ex=0, out_scale=1, return_matrix=False,dist = "normal",name = "IonotropicSynapse",verbose = True):
    """
    Initialises an excitatory inhibitory network with random weights, such that EI balance is preserved
    """

    n_units = (
        max(
            np.max(network.edges["pre_index"]),
            np.max(network.edges["post_index"]),
        )
        + 1
    )
    if verbose:
        print(n_units,out_indices,inh_indices)
    n_out = len(out_indices)
    n_units_rec = n_units-n_out
    n_inh = len(inh_indices)
    rec_inds = [ind for ind in range(n_units) if ind not in out_indices]
    recurrent_post_indices = [ind for ind in network.edges["post_index"] if ind not in out_indices]
    n_conn_rec = len(recurrent_post_indices)
    n_conn = len(network.edges["pre_index"])

    average_in_rec = n_conn_rec / n_units_rec
    #print(n_inh,n_units_rec)
    p_inh = n_inh / n_units_rec
    n_ex = n_units_rec - n_inh
    p_rec = average_in_rec / n_units_rec
    conn_matrix = np.zeros((n_units, n_units))
    EIratio = (1 - p_inh) / (p_inh)
    normaliser = np.sqrt((1 / (1 - (2 * p_rec) / np.pi)) / EIratio)

    if verbose: 
        print(average_in_rec)
        print(n_conn_rec)
        print(n_units_rec)
        print("conn probability recurrence: " + str(p_rec))
        print("EIratio:" + str(EIratio))
        print("Normaliser: " + str(normaliser))
    # this normaliser scales the variances of the two half normal distributions to be gain**2 / N

    for i in range(n_conn):
        pre = int(network.edges.iloc[i]["pre_index"])
        post = int(network.edges.iloc[i]["post_index"])
        if dist=="normal":
            samp = abs(np.random.normal(0, 1, 1)[0])
        elif dist =="uniform":
            samp = abs(np.random.uniform(0, 1))
        else:
            print("dist not recognised, use normal or uniform")
        if post in out_indices:
            samp*=out_scale
        # USE this:
        #net.select(edges=[0, 1]).set("IonotropicSynapse_gS", 0.0004)  # nS

        if pre in inh_indices:
            w = samp * normaliser * gain * EIratio / np.sqrt(average_in_rec)
            network.edges.iloc[
                i, network.edges.columns.get_loc(name+"_gS")
            ] = w
            network.edges.iloc[i, network.edges.columns.get_loc(name+"_e_syn")] = e_syn_inh
            conn_matrix[pre, post] = w * -1

        else:
            w = samp * normaliser * gain / np.sqrt(average_in_rec)
            network.edges.iloc[
                i, network.edges.columns.get_loc(name+"_gS")
            ] = w
            network.edges.iloc[i, network.edges.columns.get_loc(name+"_e_syn")] = e_syn_ex
            conn_matrix[pre, post] = w

    ev = np.linalg.eigvals(conn_matrix[rec_inds][:,rec_inds])
    if verbose:
        print("Spectral radius recurrence: " + str(np.max(np.abs(ev))))
        print("Expected gain: " + str(gain))
    # this should be close to gain!

    if return_matrix:
        return conn_matrix