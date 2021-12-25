import collections
import numpy as np

from tokenizer import Tokenizer

proteins_initials = 'ACDEFGHIKLMNPQRSTVWY'


def parse_letters(tokenizer):
    letters_or_states = collections.OrderedDict()

    while tokenizer.not_empty():
        current = tokenizer.advance()
        if current == ';':  # end of letters_or_states
            tokenizer.go_back()
            return letters_or_states
        if tokenizer.advance() != ':':  # not letters_or_states
            tokenizer.go_back()
            tokenizer.go_back()
            return None
        letters_or_states[current] = float(tokenizer.advance())


def parse_list(tokenizer):
    hypo_list = []
    while tokenizer.not_empty():
        arg = tokenizer.advance()
        if arg == ';':
            tokenizer.go_back()
            return hypo_list
        hypo_list.append(arg)


def parse_state(tokenizer):
    state_name = tokenizer.advance()
    tokenizer.advance()

    state = {}
    while tokenizer.not_empty():
        arg = tokenizer.advance()
        if arg == '}':  # end of state
            return state_name, state
        if arg in ['trans', 'only']:
            val = parse_letters(tokenizer)
            if val is None:  # a list
                val = parse_list(tokenizer)
        elif arg in ['type', 'end']:
            val = int(tokenizer.advance())
        else:
            val = tokenizer.advance()
        state[arg] = val
        tokenizer.advance()


def parse_states(tokenizer):
    states = {}
    while tokenizer.not_empty():
        name, state = parse_state(tokenizer)
        states[name] = state
    return states


def states_to_tables(states):
    states_dict = {state: index for index, state in enumerate(states.keys())}
    proteins_dict = {letter: index for index, letter in enumerate(proteins_initials)}

    transitions = np.zeros((len(states), len(states)))
    emissions = np.zeros((len(states) - 1, len(proteins_initials)))
    labels = {}
    for from_state, state_info in states.items():
        if 'label' in state_info:
            labels[from_state] = state_info['label']
        for to_state, p in state_info['trans'].items():
            transitions[states_dict[from_state], states_dict[to_state]] = p
        if from_state == 'begin':
            continue
        for letter, p in state_info['only'].items():
            emissions[states_dict[from_state] - 1, proteins_dict[letter]] = p
    np.savetxt('transition.tsv',
               np.c_[[r'state\state'] + list(states.keys()), np.vstack([list(states.keys()), transitions])],
               delimiter='\t', fmt='%s')
    np.savetxt('emissions.tsv', np.c_[list(states.keys()), np.vstack([list(proteins_initials), emissions])],
               delimiter='\t', fmt='%s')


def complete_states(states):
    for name, state in states.items():
        if 'tied_trans' in state:  # add trans from tied_trans
            states[name]['trans'] = dict(zip(state['trans'], states[state['tied_trans']]['trans'].values()))
        if 'tied_letter' in state:  # take trans from tied_letter
            states[name]['only'] = dict(states[state['tied_letter']]['only'])
    return states


def parse(lines):
    lines = ''.join(filter(lambda line: not line == '\n' and not line.startswith('#'),
                           lines[9:]))  # Get rid of empty lines, comments and header
    tokenizer = Tokenizer(lines)
    states = parse_states(tokenizer)
    states = complete_states(states)
    states_to_tables(states)


if __name__ == '__main__':
    with open('model', 'r') as model:
        model_lines = model.readlines()
    parse(model_lines)
