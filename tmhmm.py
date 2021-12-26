import pandas as pd
import numpy as np


class TMHMM:

    def __init__(self, emission_df, transition_df, alpha):
        self.aa = list(emission_df.columns)
        self.aa_dict = {letter: index for index, letter in enumerate(AA_initials)}
        self.transition = transition_df
        self.emission = emission_df
        self.states = transition_df.columns
        self.ind_dict = {}
        self.motif_length = 21
        self.alpha = alpha

    def add_end_prob_to_transition(self):
        self.transition[self.ind_dict["in_glob"], ["end"]] = self.alpha
        self.transition[self.ind_dict["out_glob"], ["end"]] = self.alpha
        s_ing = np.sum(self.transition[self.ind_dict["in_glob"], ])
        s_oug = np.sum(self.transition[self.ind_dict["out_glob"], ])
        self.transition[self.ind_dict["in_glob"], ] = self.transition[self.ind_dict["in_glob"], ] / s_ing
        self.transition[self.ind_dict["out_glob"],] = self.transition[self.ind_dict["in_glob"],] / s_oug
        return self.transition

    def create_ind_dict(self):
        ind_dict = {self.states[i]: i for i in range(len(self.states))}
        ind_dict.update({i: self.states[i]} for i in range(len(self.states)))
        self.ind_dict = ind_dict
        return self.ind_dict


if __name__ == '__main__':
    transition_path = r"C:\Users\user\PycharmProjects\CBIO_Hackathon\transition.tsv"
    emission_path = r"C:\Users\user\PycharmProjects\CBIO_Hackathon\emissions.tsv"
    transition_df = pd.read_csv(transition_path, sep="\t", index_col="state/AA")
    emission_df = pd.read_csv(emission_path, sep="\t", index_col="state/AA")





