import numpy as np
from scipy.special import logsumexp

np.seterr(divide="ignore")
AA_initials = 'ACDEFGHIKLMNPQRSTVWY^$'
AA_dict = {letter: index for index, letter in enumerate(AA_initials)}


def print_func(states, seq, motif_len):
    """
    prints the results of viterbi and posterior
    :param motif_len: motif length
    :param states: the states we got from posterior or viterbi encoding
    :param seq: the input sequence
    :return: prints on screen according to the format given.
    """
    states_lst = ["T" if 2 <= i < (motif_len * 2) + 2 else "O" for i in states]
    states_str = "".join(states_lst)
    for i in range(0, len(seq), 50):
        line_end = min(i + 50, len(seq))
        print(states_str[i:line_end])
        print(seq[i:line_end])
        print()


def forward(seq, emission_table, transition_table, motif_len):
    F = np.log(np.zeros(((motif_len * 2) + 4, len(seq)), dtype=float))
    T = np.log(transition_table)
    E = np.log(emission_table)
    F[0][0] = np.log(1)
    for col in range(1, len(seq)):
        for row in range((motif_len * 2) + 4):
            log_sum = logsumexp(F.T[col - 1] + T.T[row])
            F[row][col] = log_sum + E[row][AA_dict[seq[col]]]
    return F


def backward(seq, emission_table, transition_table, motif_len):
    B = np.log(np.ones((motif_len + 4, len(seq)), dtype=float))
    T = np.log(transition_table)
    E = np.log(emission_table)
    for col in range(len(seq) - 2, -1, -1):
        for row in range((motif_len * 2) + 4):
            B[row][col] = logsumexp(B.T[col + 1] + T[row] + E.T[AA_dict[seq[col + 1]]])
    return B


def posterior(seq, emission_table, transition_table, motif_len):
    B = backward(seq, emission_table, transition_table, motif_len)
    F = forward(seq, emission_table, transition_table, motif_len)
    P = F + B  # log space
    return np.argmax(P, axis=0)