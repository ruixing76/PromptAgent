# define task prompts for various datasets
from .base_task import BaseDataset, BaseTask
import re
import numpy as np
from typing import List, Set

# REASON labels
REASON_LABELS = {
    'helpfulAddressesClaim': 'a',
    'helpfulClear': 'b',
    'helpfulEmpathetic': 'c',
    'helpfulGoodSources': 'd',
    'helpfulImportantContext': 'e',
    'helpfulInformative': 'f',
    'helpfulUnbiasedLanguage': 'g',
    'helpfulUniqueContext': 'h',
    'notHelpfulArgumentativeOrBiased': 'i',
    'notHelpfulHardToUnderstand': 'j',
    'notHelpfulIncorrect': 'k',
    'notHelpfulIrrelevantSources': 'l',
    'notHelpfulMissingKeyPoints': 'm',
    'notHelpfulNoteNotNeeded': 'n',
    'notHelpfulOffTopic': 'o',
    'notHelpfulOpinionSpeculation': 'p',
    'notHelpfulOpinionSpeculationOrBias': 'q',
    'notHelpfulOther': 'r',
    'notHelpfulSourcesMissingOrUnreliable': 's',
    'notHelpfulSpamHarassmentOrAbuse': 't',
}

# Transform data to have one entry per claim-note pair
def flatten_data(data):
    flattened_data=[]
    for item in data:
        claim = item["claim"]
        for note in item["notes"]:
            flattened_data.append({
                "claim": claim,
                "note_text": note["text"],
                "reasons": note.get("reasons", ""),
                "label": note["label"]  # Use the label associated with the note
            })
    return flattened_data

class CustomTask(BaseTask):
    def __init__(self, 
                 train_size, 
                 eval_size,
                 test_size=None,  
                 
                 task_name = "community_notes",
                 task_description = "task from community notes",
                 data_dir='',  
                 seed=None, 
                 
                 post_instruction=True, 
                 TaskDataset=BaseDataset,
                 option_num=5, 
                 **kwargs):
        self.options = {}
        super().__init__(
                        task_name = task_name,  
                        task_description = task_description, 
                        data_dir=data_dir,
                        seed = seed,
                        train_size = train_size,
                        eval_size=eval_size,
                        test_size = test_size,
                        post_instruction = post_instruction,
                        TaskDataset=TaskDataset,
                        option_num=option_num,
                        )
        
    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        # see agent.py Line 70-71, if task_name is bigbench, it will search for and replace task_name with actual json file name (without .json)
        # I combined community_notes train, eval and test into one json file, so I need to load the whole file
        json_data = self._load_json_file(data_dir)
        self.task_description = "Predict helpfulness reason of the note in explaining the claim"
        return json_data
    
    def transform_format(self, data):
        data_split= ['train', 'eval', 'test']

        examples = {}
        for split in data_split:
            split_examples=[]
            for example in data[split]:
                task_prefix = "Predict two reasons separated by semicolon why the note is helpful or not helpful in explaining the claim using the following options"
                question_format = "Claim: {claim}\nNote: {note}\n"
                question = task_prefix+"\n"+question_format.format(claim=example['claim'], note=example['note_text'])
                # answer = set(REASON_LABELS[reason] for reason in example['reasons'].split(';'))
                answer = example['reasons']  # Use the label associated with the note

                self.options = REASON_LABELS
                options = [f'({option}) {reason}' for reason, option in REASON_LABELS.items()]
                options_str = 'Options:\n'+'\n'.join(options)
                question_str = question+'\n'+options_str+'\n'
                
                # Formatting the output
                formatted_example = {
                    'question': question_str,
                    'answer': answer
                }
                split_examples.append(formatted_example)
            examples[split] = split_examples
        return examples
    
    def clean_response(self,response):
        letters = ''.join(REASON_LABELS.values())
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        match = re.findall(clean_pattern, response.lower())
        if len(match) == 0 or not match[-1].strip():
            pattern_str = '|'.join([re.escape(option) for option in REASON_LABELS])
            backup_match = re.findall(pattern_str, response, re.IGNORECASE)

            if backup_match:
                return REASON_LABELS[backup_match[-1].lower()]
            else:
                return 'N/A: Format error'

        # Extract all valid option letters (upper or lower case), separated by semicolon, comma, or whitespace
        answer_section = match[-1]
        found_letters = re.findall(r"[" + letters + "]", answer_section)
        if not found_letters:
            return 'N/A: Format error'
        # Take at most two letters, uppercase, join with semicolon
        # result = ';'.join([l.lower() for l in found_letters[:2]])
        result = set(found_letters)
        return result
    
    def cal_correct(self, preds, labels):
        '''
        <task specific>
        The function of comparing the predictions and labels in community notes task, input are list of sets.

        preds: List of sets, each set contains the predicted labels for a claim-note pair.
        labels: List of sets, each set contains the true labels for a claim-note pair.
        Returns a list of integers, where 1 indicates a correct prediction and 0 indicates an incorrect prediction.
        '''
        comparisons = []
        for p, l in zip(preds, labels):
            # compute the intersection of predicted and true labels, comparison = intersection of p and l // union of p and l
            intersection = p.intersection(l)
            union = p.union(l)
            if len(union) == 0:
                # if both p and l are empty, we consider it a correct prediction
                comparisons.append(1)
            elif len(intersection) > 0:
                # if the intersection is not empty, it means the prediction is correct
                comparisons.append(1) # this is for gradient descent rewarding
            else:
                # if the intersection is empty, it means the prediction is incorrect
                comparisons.append(0)
        return comparisons
    
    def cal_metric(self, preds, labels, questions=None):
        '''
        <task specific>
        Calculate the evaluation metric, e.g. Accuracy, F1 score.
        "question" is for NCBI calculating F1 score.
        return a number / tuple of metrics
        
        This function is for calculating the reward of MCTS.
        '''
        correct = self.cal_correct(preds=preds, labels=labels)
        return np.mean(correct)
    
    def clean_labels(self,labels):
        '''
        <task specific>
        Transfer the form of the task ground-truth answers to List(set) 
        or List(str) that fit the input requirement of function "cal_correct"
        
        Do nothing if the data is alreadly loaded that way.
        '''
        # turn labels that separated by semicolon into a set, according to REASON_LABELS
        cleaned_labels = []
        for label in labels:
            reason_1=REASON_LABELS[label.split(';')[0]]
            reason_2=REASON_LABELS[label.split(';')[1]]
            cleaned_labels.append(set([reason_1, reason_2]))
        return cleaned_labels