import pandas as pd
import numpy as np
from scipy.special import logsumexp

np.seterr(divide="ignore")


class TMHMM:

    def __init__(self, emission_df, transition_df, alpha):
        self.aa = list(emission_df.columns)
        self.aa_dict = {letter: index for index, letter in enumerate(self.aa)}
        self.alpha = alpha
        self.transition = np.array(transition_df, dtype=float)
        self.emission = np.array(emission_df, dtype=float)
        self.log_E = np.log(self.emission)
        self.states = transition_df.columns
        self.ind_dict = self.create_ind_dict()
        self.motif_length = 21
        self.cap_length = 5
        self.num_states = len(self.states)
        self.transition = self.add_end_prob_to_transition()
        self.log_T = np.log(self.transition)
        self.non_motif_states = [self.ind_dict[state] for state in ["start", "end", "in_glob", "out_glob"]]

    def add_end_prob_to_transition(self):
        self.transition[self.ind_dict["in_glob"], self.ind_dict["end"]] = self.alpha
        self.transition[self.ind_dict["out_glob"], self.ind_dict["end"]] = self.alpha
        s_ing = np.sum(self.transition[self.ind_dict["in_glob"]])
        s_oug = np.sum(self.transition[self.ind_dict["out_glob"]])
        self.transition[self.ind_dict["in_glob"],] = self.transition[self.ind_dict["in_glob"],] / s_ing
        self.transition[self.ind_dict["out_glob"],] = self.transition[self.ind_dict["out_glob"],] / s_oug
        t = pd.DataFrame(self.transition)
        t.to_csv("tdf.csv", index=False)
        return self.transition

    def create_ind_dict(self):
        ind_dict = {self.states[i]: i for i in range(len(self.states))}
        ind_dict.update({i: self.states[i] for i in range(len(self.states))})
        self.ind_dict = ind_dict
        return self.ind_dict

    def forward(self, seq):
        F = np.log(np.zeros((self.num_states, len(seq)), dtype=float))
        F[self.ind_dict["start"]][0] = np.log(1)
        for col_num in range(1, len(seq)):
            for state in self.states:
                log_sum = logsumexp(F.T[col_num - 1] + self.log_T.T[self.ind_dict[state]])
                # if (state == "end" and seq[col_num] == "$"):
                #     print(F.T[col_num - 1], self.log_T[self.ind_dict[state]] == self.log_T.T[self.ind_dict[state]])
                F[self.ind_dict[state]][col_num] = log_sum + self.log_E[self.ind_dict[state]][
                    self.aa_dict[seq[col_num]]]
        return F

    def backward(self, seq):
        B = np.log(np.ones((self.num_states, len(seq)), dtype=float))
        for col_num in range(len(seq) - 2, -1, -1):
            for state in self.states:
                B[self.ind_dict[state]][col_num] = logsumexp(B.T[col_num + 1] + self.log_T[self.ind_dict[state]] +
                                                             self.log_E.T[self.aa_dict[seq[col_num + 1]]])
        return B

    def posterior(self, seq):
        B = self.backward(seq)
        F = self.forward(seq)
        P = F + B  # log space
        return np.argmax(P, axis=0)

    def viterbi(self, seq):
        """
        viterbi decoding - calculates the probability for each state for each position in the sequence
        :param seq: the sequence
        :param emission_table: the emission table
        :param transition_table: the transition table
        :param motif_len: the motif length
        :return: the probability for each state for each position in the sequence
        """
        V = np.log(np.zeros((self.num_states, len(seq)), dtype=float))
        Ptr = np.zeros((self.num_states, len(seq)), dtype=int)
        V[0][0] = 0
        for col_num in range(1, len(seq)):
            for state in self.states:
                V[self.ind_dict[state]][col_num] = self.log_E[self.ind_dict[state]][
                                                       self.aa_dict[seq[col_num]]] + np.max(
                    V.T[col_num - 1] + self.log_T.T[self.ind_dict[state]])
                Ptr[self.ind_dict[state]][col_num] = np.argmax(V.T[col_num - 1] + self.log_T.T[self.ind_dict[state]])

        # traceback
        states = []
        state = self.ind_dict["end"]
        for i in range(len(seq) - 1, -1, -1):
            states.append(state)
            state = Ptr[state][i]
        return states[::-1]

    def get_states_lst_for_viterbi(self, states_inds):
        lst = []
        for i in states_inds:
            if i == self.ind_dict["start"]:
                pass
            elif i == self.ind_dict["in_glob"]:
                lst.append("i")
            elif i == self.ind_dict["out_glob"]:
                lst.append("o")
            elif i == self.ind_dict["end"]:
                pass
            else:
                lst.append("M")
        return lst

    def print_func(self, states, seq):
        """
        prints the results of viterbi and posterior
        :param motif_len: motif length
        :param states: the states we got from posterior or viterbi encoding
        :param seq: the input sequence
        :return: prints on screen according to the format given.
        """
        # states_lst = ["0" if i in self.non_motif_states else "M" for i in states]
        states_lst = self.get_states_lst_for_viterbi(states)
        states_str = "".join(states_lst)
        print(states_str)
        for i in range(0, len(seq), 50):
            line_end = min(i + 50, len(seq))
            print(states_str[i:line_end])
            print(seq[i:line_end])
            print()


if __name__ == '__main__':
    transition_path = r"C:\Users\user\PycharmProjects\CBIO_Hackathon\transition_5cap.tsv"
    emission_path = r"C:\Users\user\PycharmProjects\CBIO_Hackathon\emission_5cap.tsv"
    transition_df = pd.read_csv(transition_path, sep="\t", index_col=r"state\AA")
    emission_df = pd.read_csv(emission_path, sep="\t", index_col=r"state\AA")

    tmhmm = TMHMM(emission_df, transition_df, 0.1)

    seq1 = "^MRKKILKSKVMLIALASILFVSCGGGGGGGGGGSSNLPLNPGTPSIPSTPSTPSVPEDNFPTVANPLDSQKGNISALKEK" \
            "LNRNRENSTATIPTETISYNGSTVKIGILDSDFTDPVRKAQLSARYPGIEFIPRVNSDTSTSSHGVQVLEVMMDTLEDRT" \
            "KGKAKFKAIAASIGNGGASETNKSVNPNVKTYEKVFERFNFNQKVKVVNQSFGADITIEEAPYTKNNIRNYVWAGDSKPF" \
            "ATYFEEKVNNDGGLFVWAAGNRKGATETNPGQDMDSVGMEAGLPYLVNDLEKGWIAVVGIQPKETVRVGTAPDGTPIVNI" \
            "KPNGKLNIHRTGTDRLAYAGDNAKYWSISADDSAIPTAGRAGIGSSYAAPRVSRAAALVAEKFDWMTADQVRQTLFTTTD" \
            "DTELDASLAGNANAEKRRRVKTSPDYKYGWGMLNQERALKGPGAFMDVTKYGNTNIFNAEIPAGKTSYFENKIFGFGGLV" \
            "KSGEGTLHLTNDNSYAGGSVVNRGTLEIHKIHSSKVTVNQAGRLVLHPKALIGYNEAFFNVITTVDPTRITTGTNLRNKG" \
            "IVEVNGTTAIIGGDYIAYKGSTTTFNNGAKLNVLGNIKVEDGTVKVLSDSYVTTQGSSNTVMEGKSVQGNIANVETNGMR" \
            "NANVEVQDGKVVARLSRQNPVEYIGKNAEASTKNVAENVENVFQDLDKKVMSGTATKEELAMGAIVQNMTTMGFTSATEM" \
            "MSGEIYASAQALTFSQAQNINRDLSNRLAGLDNFKNSNKDSEVWFSAIGSGGKLKRDGYASADTRVTGGQFGIDTKYKGT" \
            "TTLGVAMNYSYAKANFNRYAGESKSDMVGVSFYAKQDLPYGFYTAGRLGLSNISSKVERELLTSTGETVTGKIKHHDKML" \
            "SAYVEIGKKFGWFTPFIGYSQDYLRRGSFNESEASWGVKADRKNYRATNFLVGARAEYVGDRYKLQAYVTQAINTDKRDL" \
            "SYEGRFTGSAARQKFYGVKQSKNTTWIGFGAFREISPVFGVYGNVDFRVEDKKWADSVISTGLQYRF$"

    seq = "^MAKNLILWLVIAVVLMSVFQSFGPSESNGRKVDYSTFLQEVNNDQVREARINGREINVTKKDSNRYTTYIPVQDPKLL" \
          "DNLLTKNVKVVGEPPEEPSLLASIFISWFPMLLLIGVWIFFMRQMQGGGGKGAMSFGKSKARMLTEDQIKTTFADVAG" \
          "CDEAKEEVAELVEYLREPSRFQKLGGKIPKGVLMVGPPGTGKTLLAKAIAGEAKVPFFTISGSDFVEMFVGVGASRVRDM" \
          "FEQAKKAAPCIIFIDEIDAVGRQRGAGLGGGHDEREQTLNQMLVEMDGFEGNEGIIVIAATNRPDVLDPALLRPGRFDRQVVVGLP" \
          "DVRGREQILKVHMRRVPLAPDIDAAIIARGTPGFSGADLANLVNEAALFAARGNKRVVSMVEFEKAKDKIMMGAERRSMVMTEAQKESTAY" \
          "HEAGHAIIGRLVPEHDPVHKVTIIPRGRALGVTFFLPEGDAISASRQKLESQISTLYGGRLAEEIIYGPEHVSTGASNDIKVATNLARNMVTQWGFSE" \
          "KLGPLLYAEEEGEVFLGRSVAKAKHMSDETARIIDQEVKALIERNYNRARQLLTDNMDILHAMKDALMKYETIDAPQIDDLMARRDVRPPAGWE" \
          "EPGASNNSGDNGSPKAPRPVDEPRTPNPGNTMSEQLGDK$"

    seq2 = "^MKRLLVLCDCLWAWSLLLNALTERSYGQTSSQDELKDNTTVFTRILDRLLDGYDNRLRPGLGERVTEVKTDIFVTSFGPVSDHDMEYTIDVFFRQSWKDERLKFKGPMTVLRLNNLMASKIWTPDTFFHNGKKSVAHNMTMPNKLLRITEDGTLLYTMRLTVRAECPMHLEDFPMDVHACPLKFGSYAYTRAEVVYEWTREPARSVVVAEDGSRLNQYDLLGQTVDSGIVQSSTGEYVVMTTHFHLKRKIGYFVIQTYLPCIMTVILSQVSFWLNRESVPARTVFGVTTVLTMTTLSISARNSLPKVAYATAMDWFIAVCYAFVFSALIEFATVNYFTKRGYAWDGKSVVPEKPKKVKDPLIKKNNTYTAAATSYTPNIARDPGLATIAKSATIEPKEVKPETKPAEPKKTFNSVSKIDRLSRIAFPLLFGIFNLVYWATYLNREPQLKAPTPHQ$"

    v = tmhmm.viterbi(seq2)
    tmhmm.print_func(v, seq2[1: len(seq2) - 1])
