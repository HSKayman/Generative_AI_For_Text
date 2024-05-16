# Written by: Hasan Suca Kayman
# City College of New York, CUNY
# February 2024
# chatbot_preprocessor

# Importing necessary libraries
import re
import unicodedata

# File to be processed
file_name = 'dialogs.txt'

# Convert Unicode characters to ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

# Expand common English contractions
def expand_contractions(text):
    
    contraction_map = {
        "i'm": "I am",
        "i've": "I have",
        "it's": "it is",
        "how's": "how is",
        "everything's": "everything is",
        "haven't": "have not",
        "shouldn't": "should not",
        "wasn't": "was not",
        "can't": "cannot",
        "i'd": "I would",
        "doesn't": "does not",
        "you're": "you are",
        "wouldn't": "would not",
        "that's": "that is",
        "didn't": "did not",
        "isn't": "is not",
        "don't": "do not",
        "what's": "what is",
        "it'll": "it will",
        "what'll": "what will",
        "let's": "let us",
        "i'll": "I will",
        "she's": "she is",
        "there's": "there is",
        "might've": "might have",
        "you've": "you have",
        "weren't": "were not",
        "macy's": "Macy's (store)",
        "couldn't": "could not",
        "night's": "night is",
        "must've": "must have",
        "should've": "should have",
        "would've": "would have",
        "didn't?": "did not?",
        "mom's": "mom is",
        "they're": "they are",
        "where's": "where is",
        "here's": "here is",
        "we're": "we are",
        "you'll": "you will",
        "he'll": "he will",
        "he's": "he is",
        "won't": "will not",
        "mcdonald's": "McDonald's",
        "grandma's": "grandma is",
        "people's": "people are",
        "something's": "something is",
        "you'd": "you would",
        "aren't": "are not",
        "nothing's": "nothing is",
        "a's": "A is",
        "b's": "B is",
        "why's": "why is",
        "shoes—they're": "shoes — they are",
        "mother's": "mother is",
        "they'll": "they will",
        "when's": "when is",
        "dad's": "dad is",
        "driver's": "driver is",
        "o'clock": "of the clock",
    }
    
    
    # Expand contractions based on the mapping
    def expand(match):
        contraction = match.group(0).lower()  
        if contraction in contraction_map:
            return contraction_map[contraction]
        return contraction  
    

    pattern = re.compile(r"\b(" + "|".join(contraction_map.keys()) + r")\b", re.IGNORECASE)
    return pattern.sub(expand, text)

# Preprocess a given sentence     
def preprocess_sentence(w):
    
    w = expand_contractions(w.lower().strip())
    w = unicode_to_ascii(w)
    
    # Separate punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w) 

    # Remove extra spaces
    w = re.sub(r'[" "]+', " ", w)

    # Keep only specific characters
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # Add start and end tokens
    w = '<start> ' + w + ' <end>'
    return w

# Read the original dialog file
file = open(file_name,'r').read()
qna_list = [f.split('\t') for f in file.split('\n')][:-1]


# Extract questions and answers
questions = [x[0] for x in qna_list]
answers = [x[1] for x in qna_list]

# Preprocess the questions and answers
pre_questions = [preprocess_sentence(w) for w in questions]
pre_answers = [preprocess_sentence(w) for w in answers]

# Save the preprocessed dialog to a new file
clean_dialog = ""
for index,i in enumerate(pre_questions):
    clean_dialog+=pre_questions[index]
    clean_dialog+= "\t"
    clean_dialog+=pre_answers[index]
    clean_dialog+="\n"

file = open("cleaned" + file_name, "w")
file.write(clean_dialog)
file.close()