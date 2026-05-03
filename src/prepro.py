from tqdm import tqdm
import ujson as json


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = []
        if self.args.input_format == 'entity_marker':
            self.new_tokens = ['[E1]', '[/E1]', '[E2]', '[/E2]']
        self.tokenizer.add_tokens(self.new_tokens)
        # changes: added 'typed_entity_marker_pos', 'typed_entity_marker_pos_punct' to the list of valid formats
        if self.args.input_format not in ('entity_mask', 'entity_marker', 'entity_marker_punct', 'typed_entity_marker', 'typed_entity_marker_punct', 'typed_entity_marker_pos', 'typed_entity_marker_pos_seq', 'typed_entity_marker_pos_set', 'typed_entity_marker_pos_punct', 'typed_entity_marker_pos_seq_punct', 'typed_entity_marker_pos_set_punct', 'typed_entity_marker_deprel_set_punct'):
            raise Exception("Invalid input format!")

    def tokenize(self, tokens, subj_type, obj_type, ss, se, os, oe, pos_tags=None, deprel_tags=None):
        """
        Implement the following input formats:
            - entity_mask: [SUBJ-NER], [OBJ-NER].
            - entity_marker: [E1] subject [/E1], [E2] object [/E2].
            - entity_marker_punct: @ subject @, # object #.
            - typed_entity_marker: [SUBJ-NER] subject [/SUBJ-NER], [OBJ-NER] obj [/OBJ-NER]
            - typed_entity_marker_punct: @ * subject ner type * subject @, # ^ object ner type ^ object #
            - typed_entity_marker_pos: [SUBJ-NER*POS1] subject [/SUBJ-NER*POS1], [OBJ-NER*POS1] obj [/OBJ-NER*POS1] <-- Appends the first PoS tag to the marker
            - typed_entity_marker_pos_seq: [SUBJ-NER*POS1+POS2] subject [/SUBJ-NER*POS1+POS2], [OBJ-NER*POS1+POS2] obj [/OBJ-NER*POS1+POS2] <-- Appends all PoS tags \in multiset(entity_{POStags})
            - typed_entity_marker_pos_set: [SUBJ-NER*UNIQUEPOS1+UNIQUEPOS2] subject [/SUBJ-NER*UNIQUEPOS1+UNIQUEPOS2], [OBJ-NER*UNIQUEPOS1+UNIQUEPOS2] obj [/OBJ-NER*UNIQUEPOS1+UNIQUEPOS2] <-- Appends all PoS tags \in set(entity_{POStags})
            - typed_entity_marker_pos_punct: @ * subject ner type + pos tag * subject @, # ^ object ner type pos tag ^ object # <-- Appends first PoS tag using punct representation
            - typed_entity_marker_pos_seq_punct: @ * subject ner type + pos_1 + pos_2 * subject_{token_{1}} subject_{token_{2}} @, # ^ object ner type pos_1 ^ object # <-- Appends all PoS tags \in multiset(entity_{POStags}) using punct representation
            - typed_entity_marker_pos_set_punct: @ * subject ner type + pos_1 + pos_2 * subject_{token_{1}} subject_{token_{2}} @, # ^ object ner type pos_1 ^ object # <-- Appends all PoS tags \in multiset(entity_{POStags}) using punct representation
        """
        if not pos_tags or not deprel_tags:
            raise Exception("No pos/deprel tags supplied.");
    
        sents = []
        input_format = self.args.input_format
        
        if input_format == 'entity_mask':
            subj_type = '[SUBJ-{}]'.format(subj_type)
            obj_type = '[OBJ-{}]'.format(obj_type)
            for token in (subj_type, obj_type):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        
        elif input_format == 'typed_entity_marker':
            subj_start = '[SUBJ-{}]'.format(subj_type)
            subj_end = '[/SUBJ-{}]'.format(subj_type)
            obj_start = '[OBJ-{}]'.format(obj_type)
            obj_end = '[/OBJ-{}]'.format(obj_type)
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token)
                    self.tokenizer.add_tokens([token])
        
        # approach #1: producing a subj/obj strart & subj/obj end tag by appending the PoS tag @ ss, se, os & oe \in multiset(entity_{POStags}) to 'subject/object' & entity-type:
        # e.g. [SUBJ-PERSON-NPP] Alice Dellal [/SUBJ-PERSON-NPP]
        elif input_format == 'typed_entity_marker_pos':
            # construct subject tags
            subj_start = '[SUBJ-{}*{}]'.format(subj_type, pos_tags[ss]);
            subj_end = '[/SUBJ-{}*{}]'.format(subj_type, pos_tags[se]);
            # & object tags
            obj_start = '[OBJ-{}*{}]'.format(obj_type, pos_tags[os]);
            obj_end = '[/OBJ-{}*{}]'.format(obj_type, pos_tags[oe]);
            # if these tokens are considered unseen new tokens, add them to the set of new tokens
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token);
                    self.tokenizer.add_tokens([token]);
        
        # approach #2: producing a subj start & subj end tag by concat'ing the subject/object, entity-type & sequenced PoS tags together
        # if the entity length is > 1, then produce [SUBJ-TYPE*POS1+POS2] subj_token_1 subj_token_2 [/SUBJ-TYPE*POS1+POS2] & same for the obj
        elif input_format == 'typed_entity_marker_pos_seq':
            # grabbing the PoS between the subject start & subject end 
            subj_pos_tags = pos_tags[ss : se + 1];
            # constructing a PoS string to represent the sequence of tags
            subj_pos_string = "+".join(subj_pos_tags);
            # building the subj opening & closing tag
            subj_start = '[SUBJ-{}*{}]'.format(subj_type, subj_pos_string);
            subj_end = '[/SUBJ-{}*{}]'.format(subj_type, subj_pos_string);
            # same again for object start & end tag
            obj_pos_tags = pos_tags[os : oe + 1];
            obj_pos_string = "+".join(obj_pos_tags);
            obj_start = '[OBJ-{}*{}]'.format(obj_type, obj_pos_string);
            obj_end = '[/OBJ-{}*{}]'.format(obj_type, obj_pos_string);
            # if these tokens are considered unseen new tokens, add them to the set of new tokens
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token);
                    self.tokenizer.add_tokens([token]);

        # approach #3: similar to approach #2, but simply take the set of PoS tags
        elif input_format == 'typed_entity_marker_pos_set':
            # rather than use set(), we're going to retain some degree of order
            subj_pos_tags = list(dict.fromkeys(pos_tags[ss : se + 1]));
            subj_pos_string = "+".join(subj_pos_tags);
            subj_start = '[SUBJ-{}*{}]'.format(subj_type, subj_pos_string);
            subj_end = '[/SUBJ-{}*{}]'.format(subj_type, subj_pos_string);
            # same for obj
            obj_pos_tags = list(dict.fromkeys(pos_tags[os : oe + 1]));
            obj_pos_string = "+".join(obj_pos_tags);
            obj_start = '[OBJ-{}*{}]'.format(obj_type, obj_pos_string);
            obj_end = '[/OBJ-{}*{}]'.format(obj_type, obj_pos_string);
            for token in (subj_start, subj_end, obj_start, obj_end):
                if token not in self.new_tokens:
                    self.new_tokens.append(token);
                    self.tokenizer.add_tokens([token]);

        # original adaptation from: https://arxiv.org/pdf/2102.01373v4
        elif input_format == 'typed_entity_marker_punct':
            subj_type = self.tokenizer.tokenize(subj_type.replace("_", " ").lower())
            obj_type = self.tokenizer.tokenize(obj_type.replace("_", " ").lower())

        # approach #4: adding a single PoS tag 
        elif input_format == 'typed_entity_marker_pos_punct':
            subj_first_pos = pos_tags[ss];
            obj_first_pos = pos_tags[os];
            subj_type = subj_type.replace("_", " ").lower()
            subj_type = self.tokenizer.tokenize(subj_type + "+" + subj_first_pos)
            obj_type = obj_type.replace("_", " ").lower()
            obj_type = self.tokenizer.tokenize(obj_type + "+" + obj_first_pos)

        # approach #5:
        elif input_format == 'typed_entity_marker_pos_seq_punct':
            # see approach #2
            subj_pos_tags = pos_tags[ss : se + 1];
            subj_pos_string = "+".join(subj_pos_tags);
            obj_pos_tags = pos_tags[os : oe + 1];
            obj_pos_string = "+".join(obj_pos_tags);
            # reuse part of originally adapted approach
            subj_type = subj_type.replace("_", " ").lower();
            obj_type = obj_type.replace("_", " ").lower();
            # concatenate PoS tags
            subj_type = self.tokenizer.tokenize(subj_type) + [':'] + self.tokenizer.tokenize(subj_pos_string);
            obj_type = self.tokenizer.tokenize(obj_type) + [':'] + self.tokenizer.tokenize(obj_pos_string);

        # approach #6:
        elif input_format == 'typed_entity_marker_pos_set_punct':
            # see approach #3
            subj_pos_tags = list(dict.fromkeys(pos_tags[ss : se + 1]));
            subj_pos_string = "+".join(subj_pos_tags);
            obj_pos_tags = list(dict.fromkeys(pos_tags[os : oe + 1]));
            obj_pos_string = "+".join(obj_pos_tags);
            # reuse part of originally adapted approach
            subj_type = subj_type.replace("_", " ").lower();
            obj_type = obj_type.replace("_", " ").lower();
            # concatenate PoS tags
            subj_type = self.tokenizer.tokenize(subj_type) + [":"] + self.tokenizer.tokenize(subj_pos_string);
            obj_type = self.tokenizer.tokenize(obj_type) + [":"] + self.tokenizer.tokenize(obj_pos_string);

        # approach #7
        elif input_format == 'typed_entity_marker_deprel_set_punct':
            # see approach #6
            subj_deprel_tags = list(dict.fromkeys(deprel_tags[ss : se]));
            subj_deprel_string = " ".join(subj_deprel_tags);
            obj_deprel_tags = list(dict.fromkeys(deprel_tags[os : oe]));
            obj_deprel_string = " ".join(obj_deprel_tags);
            inj_subj_type = subj_type.replace("_", " ").lower();
            inj_obj_type = obj_type.replace("_", " ").lower();
            subj_type = self.tokenizer.tokenize(inj_subj_type) + self.tokenizer.tokenize(subj_deprel_string);
            obj_type = self.tokenizer.tokenize(inj_obj_type) + self.tokenizer.tokenize(obj_deprel_string);

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)

            if input_format == 'entity_mask':
                if ss <= i_t <= se or os <= i_t <= oe:
                    tokens_wordpiece = []
                    if i_t == ss:
                        new_ss = len(sents)
                        tokens_wordpiece = [subj_type]
                    if i_t == os:
                        new_os = len(sents)
                        tokens_wordpiece = [obj_type]

            elif input_format == 'entity_marker':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['[E1]'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['[/E1]']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['[E2]'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['[/E2]']

            elif input_format == 'entity_marker_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ['#'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ['#']

            elif input_format == 'typed_entity_marker' or input_format == 'typed_entity_marker_pos' or input_format == 'typed_entity_marker_pos_seq' or input_format == 'typed_entity_marker_pos_set':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = [subj_start] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + [subj_end]
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = [obj_start] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + [obj_end]

            elif input_format == 'typed_entity_marker_punct' or input_format == 'typed_entity_marker_pos_punct' or input_format == 'typed_entity_marker_pos_seq_punct' or input_format == 'typed_entity_marker_pos_set_punct' or input_format == 'typed_entity_marker_deprel_set_punct':
                if i_t == ss:
                    new_ss = len(sents)
                    tokens_wordpiece = ['@'] + ['*'] + subj_type + ['*'] + tokens_wordpiece
                if i_t == se:
                    tokens_wordpiece = tokens_wordpiece + ['@']
                if i_t == os:
                    new_os = len(sents)
                    tokens_wordpiece = ["#"] + ['^'] + obj_type + ['^'] + tokens_wordpiece
                if i_t == oe:
                    tokens_wordpiece = tokens_wordpiece + ["#"]

            sents.extend(tokens_wordpiece)
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids, new_ss + 1, new_os + 1


class TACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            # modified tokenize to also accept the PoS tags, it's important to understand that whilst the subj_type & obj_type are a single token, i.e. 'person' or 'organisation'
            # the PoS tags could be a list of length > 1. This may cause some issues when representing the the start-subject, end-subject, start-object & end-object as
            # their own distinct tags (i.e. [SUB-PERSON-NNP-NNP-NNP] jon anthony dil [/SUB-PERSON-NNP-NNP-NNP]) due to the creeping tag length
            # Further: using an alternative representation may shorten the observable context, i.e. '@ * person : nnp + nnp + nnp * jon anthony dil @'
            # TODO: go back and read through the papers again on the various models we're using here to understand the implications of these various changes in more depth
            input_ids, new_ss, new_os = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe, pos_tags=d['stanford_pos'], deprel_tags=d['stanford_deprel']) # <-- CHANGES
            rel = self.LABEL_TO_ID[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features


class RETACREDProcessor(Processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer)
        self.LABEL_TO_ID = {'no_relation': 0, 'org:founded_by': 1, 'per:identity': 2, 'org:alternate_names': 3, 'per:children': 4, 'per:origin': 5, 'per:countries_of_residence': 6, 'per:employee_of': 7, 'per:title': 8, 'org:city_of_branch': 9, 'per:religion': 10, 'per:age': 11, 'per:date_of_death': 12, 'org:website': 13, 'per:stateorprovinces_of_residence': 14, 'org:top_members/employees': 15, 'org:number_of_employees/members': 16, 'org:members': 17, 'org:country_of_branch': 18, 'per:spouse': 19, 'org:stateorprovince_of_branch': 20, 'org:political/religious_affiliation': 21, 'org:member_of': 22, 'per:siblings': 23, 'per:stateorprovince_of_birth': 24, 'org:dissolved': 25, 'per:other_family': 26, 'org:shareholders': 27, 'per:parents': 28, 'per:charges': 29, 'per:schools_attended': 30, 'per:cause_of_death': 31, 'per:city_of_death': 32, 'per:stateorprovince_of_death': 33, 'org:founded': 34, 'per:country_of_death': 35, 'per:country_of_birth': 36, 'per:date_of_birth': 37, 'per:cities_of_residence': 38, 'per:city_of_birth': 39}

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)

        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [convert_token(token) for token in tokens]

            # modified tokenize to also accept the PoS tags
            input_ids, new_ss, new_os = self.tokenize(tokens, d['subj_type'], d['obj_type'], ss, se, os, oe, pos_tags=d['stanford_pos'], deprel_tags=d['stanford_deprel']) # <-- CHANGES
            rel = self.LABEL_TO_ID[d['relation']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }

            features.append(feature)
        return features
