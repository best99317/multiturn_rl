import os
import os.path as osp

current_dir = osp.dirname(__file__)

TERMINATION_SIGNAL = os.getenv('TERMINATION_SIGNAL', "[[TERMINATE CHAT]]")

with open(osp.join(current_dir, 'test_assistant_prompt.txt'), 'r') as f:
    ASSITANT_PROMPT = f.read()

with open(osp.join(current_dir, 'test_user_prompt.txt'), 'r') as f:
    USER_PROMPT = f.read()