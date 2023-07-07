from PIL import Image
from demo_init import init_task, ask_answer, get_examples
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

"""
Example instructions:
report_inst = " what can we get from this chest medical image? "
vg_inst = ' which region does the text "edema" describe? '
vqa_inst = 'what disease does the image show?'
"""

def run(demo_image, demo_instruction='what can we get from this chest medical image?'):
    init_task()
    if len(demo_image) == 0:
        demo_data = get_examples()[0]
        demo_image = demo_data[0]
        
    img = Image.open(demo_image).convert('RGB')
    demo_instruction = demo_data[1].split('&&')[0]
    result = ask_answer(img, demo_instruction)
    return 


if __name__ == '__main__':
    demo_image = ''
    run(demo_image)