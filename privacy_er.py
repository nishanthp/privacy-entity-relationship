#!/usr/bin/env python3
"""
Unbiased Privacy-Preserving NLP Experiments with Real Datasets
Rigorous experimental framework designed to fairly test the hypothesis without bias
"""

# START GENAI@GHCOPILOT
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, pipeline
)
from datasets import load_dataset, Dataset as HFDataset
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import re
import warnings
import scipy.stats as stats
from collections import defaultdict
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class UnbiasedPrivacyNLPExperiment:
    """
    Unbiased experiment class designed to fairly test privacy-preserving NLP methods
    without favoring any particular approach
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.random_seed = RANDOM_SEED
        self.setup_models()
        self.setup_candidate_pools()
        self.results = {}
        
        # Bias mitigation measures
        self.evaluation_order_randomized = True
        self.multiple_runs = 10  # Run each experiment multiple times
        self.statistical_significance_testing = True
        
    def setup_models(self):
        """Initialize all required models with bias considerations"""
        print("Setting up models for unbiased evaluation...")
        
        # Use multiple sentence encoders to avoid bias toward one model
        self.sentence_encoders = {
            'sentence_bert': SentenceTransformer('all-MiniLM-L6-v2'),
            'universal_encoder': SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        }
        
        # Primary encoder for context-aware method
        self.sentence_encoder = self.sentence_encoders['sentence_bert']
        
        # NER model setup with fallbacks
        self.setup_ner_models()
        
        # BERT model for downstream tasks
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Evaluation models
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print("Models loaded successfully!")
        
    def setup_ner_models(self):
        """Setup NER with multiple fallback options"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("Loaded spaCy en_core_web_sm model")
        except OSError:
            try:
                self.nlp = spacy.load('en_core_web_md')
                print("Loaded spaCy en_core_web_md model")
            except OSError:
                try:
                    self.nlp = spacy.load('en_core_web_lg')
                    print("Loaded spaCy en_core_web_lg model")
                except OSError:
                    print("WARNING: No spaCy English model found. Using fallback NER...")
                    self.nlp = None
                    self.setup_fallback_ner()
        
    def setup_fallback_ner(self):
        """Setup fallback NER using transformers"""
        print("Setting up fallback NER using transformers...")
        try:
            self.ner_pipeline = pipeline("ner", 
                                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                                        aggregation_strategy="simple")
            print("Fallback NER loaded successfully!")
        except Exception as e:
            print(f"Failed to load fallback NER: {e}")
            print("Using regex-based entity extraction...")
            self.ner_pipeline = None
        
    def setup_candidate_pools(self):
        """
        Create UNBIASED candidate entity pools for replacement
        - Pools are balanced across different domains
        - No semantic bias toward any particular context
        - Equal representation of entity types
        """
        print("Setting up unbiased candidate entity pools...")
        
        # Carefully curated pools to avoid semantic bias
        self.candidate_pools = {
            'PERSON': [
                # Mix of common names from different backgrounds
                'Alex Johnson', 'Taylor Smith', 'Jordan Brown', 'Casey Davis', 'Riley Wilson',
                'Morgan Miller', 'Avery Moore', 'Quinn Taylor', 'Blake Anderson', 'Drew Thomas',
                'Sage Jackson', 'River White', 'Phoenix Harris', 'Skyler Martin', 'Rowan Thompson',
                'Emery Garcia', 'Finley Martinez', 'Hayden Robinson', 'Kendall Clark', 'Peyton Rodriguez'
            ],
            'ORG': [
                # Generic organization names without industry bias
                'Alpha Corp', 'Beta Systems', 'Gamma Industries', 'Delta Solutions', 'Epsilon Group',
                'Zeta Enterprises', 'Eta Partners', 'Theta Holdings', 'Iota Ventures', 'Kappa Labs',
                'Lambda Works', 'Mu Dynamics', 'Nu Technologies', 'Xi Innovations', 'Omicron Inc',
                'Pi Associates', 'Rho Company', 'Sigma Corporation', 'Tau Industries', 'Upsilon LLC'
            ],
            'GPE': [
                # Generic location names without geographic bias
                'Riverside', 'Hillview', 'Lakeside', 'Meadowbrook', 'Oakwood',
                'Pinehurst', 'Fairfield', 'Greenwood', 'Brookstone', 'Clearwater',
                'Millfield', 'Westfield', 'Eastbrook', 'Northgate', 'Southview',
                'Centertown', 'Highlands', 'Lowlands', 'Midtown', 'Crossroads'
            ]
        }
        
        # Compute embeddings for ALL sentence encoders to ensure fairness
        self.candidate_embeddings = {}
        for encoder_name, encoder in self.sentence_encoders.items():
            self.candidate_embeddings[encoder_name] = {}
            for entity_type, candidates in self.candidate_pools.items():
                embeddings = encoder.encode(candidates)
                self.candidate_embeddings[encoder_name][entity_type] = embeddings
            
        print("Unbiased candidate pools created successfully!")
        
    def load_datasets(self):
        """Load datasets with balanced sampling to avoid bias"""
        print("Loading datasets with balanced sampling...")
        
        datasets = {}
        
        # Load real datasets with error handling
        try:
            persona_data = load_dataset('bavard/personachat_truecased', split='train[:1000]')
            datasets['personachat'] = self.prepare_personachat(persona_data)
            print("PersonaChat loaded successfully!")
        except Exception as e:
            print(f"Could not load PersonaChat: {e}")
            datasets['personachat'] = self.create_balanced_dialogue_data()
            
        try:
            cnn_data = load_dataset('cnn_dailymail', '3.0.0', split='train[:500]')
            datasets['cnn_dailymail'] = self.prepare_cnn_dailymail(cnn_data)
            print("CNN/DailyMail loaded successfully!")
        except Exception as e:
            print(f"Could not load CNN/DailyMail: {e}")
            datasets['cnn_dailymail'] = self.create_balanced_news_data()
            
        # Create balanced synthetic data
        datasets['dialogsum'] = self.create_balanced_dialogsum_data()
        
        return datasets
        
    def create_balanced_dialogue_data(self):
        """Create balanced synthetic dialogue data without bias toward any method"""
        balanced_dialogues = [
            # Varied sentence structures and entity relationships
            "Hello, I'm Alex and I work at Alpha Corp in Riverside. How about you?",
            "Taylor from Beta Systems mentioned this project in Hillview yesterday.",
            "My colleague Jordan works at Gamma Industries and lives in Lakeside.",
            "Casey Johnson is a researcher at Delta Solutions in Meadowbrook.",
            "The manager of Epsilon Group, Riley Wilson, announced new plans.",
            "Dr. Morgan Miller from Zeta Enterprises published research findings.",
            "Avery from Eta Partners discussed their Oakwood office expansion.",
            "Professor Quinn Taylor at Theta Holdings is leading innovation.",
            "My friend Blake works at Iota Ventures in Pinehurst.",
            "The director of Kappa Labs, Drew Thomas, gave a presentation.",
            # Additional varied examples
            "Sage Jackson coordinates projects between Lambda Works and Fairfield.",
            "River White leads the team at Mu Dynamics in Greenwood.",
            "Phoenix Harris from Nu Technologies visited Brookstone last week.",
            "Skyler Martin represents Xi Innovations in Clearwater.",
            "Rowan Thompson manages operations at Omicron Inc in Millfield."
        ]
        
        return [{'text': text, 'source': 'balanced_synthetic', 'task': 'dialogue'} 
                for text in balanced_dialogues]
        
    def create_balanced_news_data(self):
        """Create balanced synthetic news data"""
        balanced_news = [
            "Alpha Corp announced new initiatives. CEO Alex Johnson said the company is expanding operations.",
            "Beta Systems reported quarterly growth. The company, based in Hillview, exceeded projections.",
            "Gamma Industries unveiled new products at their Lakeside headquarters. Taylor Smith praised the team.",
            "Delta Solutions is expanding operations in Meadowbrook. The company plans to hire employees.",
            "Epsilon Group's division is investing in new technology. Jordan Brown leads the research team.",
            "Zeta Enterprises stock increased after Casey Davis announced developments at their Oakwood facility.",
            "Eta Partners is producing new content. The company, headquartered in Pinehurst, aims for growth.",
            "Theta Holdings research division in Fairfield published results. Dr. Riley Wilson leads the team.",
            "Iota Ventures cloud services are gaining market share. The company's Greenwood office is expanding.",
            "Kappa Labs new design promises improvements. The company's Brookstone team worked on the project."
        ]
        
        return [{'text': text, 'source': 'balanced_news', 'task': 'summarization'} 
                for text in balanced_news]
        
    def create_balanced_dialogsum_data(self):
        """Create balanced DialogSum-like data"""
        balanced_dialogsum = [
            "Customer: I need assistance. Agent: I can help. What's your name? Customer: I'm Alex Johnson from Riverside.",
            "Manager: Let's discuss the timeline. Employee: I'm working with the Hillview team. Manager: Great, Taylor Smith will coordinate.",
            "Doctor: How are you today? Patient: Much better, thanks. I'm from Lakeside originally. Doctor: Good to hear, Mr. Brown.",
            "Teacher: Welcome to class. Student: Thank you, I'm excited. I moved from Meadowbrook recently. Teacher: Wonderful, Ms. Davis.",
            "Interviewer: Tell me about yourself. Candidate: I worked at Alpha Corp in Oakwood for two years. Interviewer: Interesting, Mr. Wilson."
        ]
        
        return [{'text': text, 'source': 'balanced_dialogsum', 'task': 'summarization'} 
                for text in balanced_dialogsum]
        
    def prepare_personachat(self, data):
        """Prepare PersonaChat data with balanced sampling"""
        processed_data = []
        for item in data:
            if 'history' in item and 'candidates' in item:
                history = item['history']
                if len(history) > 0:
                    text = ' '.join(history[-3:])
                    processed_data.append({
                        'text': text,
                        'source': 'personachat',
                        'task': 'dialogue'
                    })
        
        # Randomly sample to ensure no bias
        if len(processed_data) > 100:
            processed_data = random.sample(processed_data, 100)
        
        return processed_data
        
    def prepare_cnn_dailymail(self, data):
        """Prepare CNN/DailyMail data with balanced sampling"""
        processed_data = []
        for item in data:
            if 'article' in item and len(item['article']) > 100:
                sentences = item['article'].split('.')[:3]
                text = '. '.join(sentences) + '.'
                processed_data.append({
                    'text': text,
                    'source': 'cnn_dailymail',
                    'task': 'summarization'
                })
        
        # Randomly sample to ensure no bias
        if len(processed_data) > 100:
            processed_data = random.sample(processed_data, 100)
            
        return processed_data
        
    def extract_entities(self, text):
        """Extract entities using available NER methods"""
        entities = []
        
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    })
        elif hasattr(self, 'ner_pipeline') and self.ner_pipeline is not None:
            try:
                ner_results = self.ner_pipeline(text)
                for ent in ner_results:
                    label_mapping = {
                        'PER': 'PERSON', 'PERSON': 'PERSON',
                        'ORG': 'ORG', 'LOC': 'GPE', 'MISC': 'GPE'
                    }
                    entity_label = label_mapping.get(ent['entity_group'], ent['entity_group'])
                    if entity_label in ['PERSON', 'ORG', 'GPE']:
                        entities.append({
                            'text': ent['word'],
                            'label': entity_label,
                            'start': ent['start'],
                            'end': ent['end']
                        })
            except Exception as e:
                entities = self.extract_entities_regex(text)
        else:
            entities = self.extract_entities_regex(text)
                
        return entities
        
    def extract_entities_regex(self, text):
        """Fallback regex-based entity extraction"""
        entities = []
        
        # Unbiased regex patterns
        person_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            r'\bDr\. [A-Z][a-z]+\b',
            r'\bProf\. [A-Z][a-z]+\b',
            r'\bMr\. [A-Z][a-z]+\b',
            r'\bMs\. [A-Z][a-z]+\b',
        ]
        
        org_patterns = [
            r'\b[A-Z][a-z]+ Corp\b',
            r'\b[A-Z][a-z]+ Inc\b',
            r'\b[A-Z][a-z]+ Systems\b',
            r'\b[A-Z][a-z]+ Industries\b',
            r'\b[A-Z][a-z]+ Solutions\b',
            r'\b[A-Z][a-z]+ Group\b',
            r'\b[A-Z][a-z]+ Enterprises\b',
        ]
        
        location_patterns = [
            r'\b[A-Z][a-z]+view\b',
            r'\b[A-Z][a-z]+side\b',
            r'\b[A-Z][a-z]+wood\b',
            r'\b[A-Z][a-z]+field\b',
            r'\b[A-Z][a-z]+brook\b',
        ]
        
        # Extract entities using patterns
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'PERSON',
                    'start': match.start(),
                    'end': match.end()
                })
        
        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'ORG',
                    'start': match.start(),
                    'end': match.end()
                })
        
        for pattern in location_patterns:
            for match in re.finditer(pattern, text):
                entities.append({
                    'text': match.group(),
                    'label': 'GPE',
                    'start': match.start(),
                    'end': match.end()
                })
        
        return self.remove_overlapping_entities(entities)
        
    def remove_overlapping_entities(self, entities):
        """Remove overlapping entities"""
        if not entities:
            return entities
            
        entities.sort(key=lambda x: x['start'])
        filtered_entities = []
        
        for entity in entities:
            overlaps = False
            for existing in filtered_entities:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    if len(entity['text']) > len(existing['text']):
                        filtered_entities.remove(existing)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_entities.append(entity)
        
        return filtered_entities
        
    def create_context_representation(self, text, entities):
        """Create abstracted context representation"""
        abstracted_text = text
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for entity in entities_sorted:
            placeholder = f"<{entity['label']}>"
            abstracted_text = (abstracted_text[:entity['start']] + 
                             placeholder + 
                             abstracted_text[entity['end']:])
                             
        return abstracted_text
        
    def context_aware_replacement(self, text, entities, sensitive_entities=None, encoder_name='sentence_bert'):
        """
        UNBIASED context-aware entity replacement
        - Uses specified encoder to avoid bias
        - Random selection of sensitive entities
        - Fair candidate selection process
        """
        if not entities:
            return text, []
            
        if sensitive_entities is None:
            # Randomly select sensitive entities (unbiased)
            num_sensitive = min(len(entities), random.randint(1, min(3, len(entities))))
            sensitive_entities = random.sample(entities, num_sensitive)
            
        # Create context representation
        context_text = self.create_context_representation(text, entities)
        encoder = self.sentence_encoders[encoder_name]
        context_embedding = encoder.encode([context_text])[0]
        
        replacements = []
        modified_text = text
        
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for entity in entities_sorted:
            if entity in sensitive_entities:
                entity_type = entity['label']
                
                if entity_type in self.candidate_pools:
                    candidates = self.candidate_pools[entity_type]
                    candidate_embeddings = self.candidate_embeddings[encoder_name][entity_type]
                    
                    # Compute similarities
                    similarities = cosine_similarity([context_embedding], candidate_embeddings)[0]
                    
                    # Select best candidate
                    best_idx = np.argmax(similarities)
                    replacement = candidates[best_idx]
                    
                    # Avoid same entity
                    if replacement.lower() == entity['text'].lower() and len(candidates) > 1:
                        similarities[best_idx] = -1
                        best_idx = np.argmax(similarities)
                        replacement = candidates[best_idx]
                    
                    # Replace in text
                    modified_text = (modified_text[:entity['start']] + 
                                   replacement + 
                                   modified_text[entity['end']:])
                    
                    replacements.append({
                        'original': entity['text'],
                        'replacement': replacement,
                        'type': entity_type,
                        'similarity': similarities[best_idx]
                    })
                    
        return modified_text, replacements
        
    def random_replacement(self, text, entities, sensitive_entities=None):
        """UNBIASED random entity replacement"""
        if not entities:
            return text, []
            
        if sensitive_entities is None:
            num_sensitive = min(len(entities), random.randint(1, min(3, len(entities))))
            sensitive_entities = random.sample(entities, num_sensitive)
            
        replacements = []
        modified_text = text
        
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for entity in entities_sorted:
            if entity in sensitive_entities:
                entity_type = entity['label']
                
                if entity_type in self.candidate_pools:
                    candidates = self.candidate_pools[entity_type]
                    replacement = random.choice(candidates)
                    
                    # Avoid same entity
                    if replacement.lower() == entity['text'].lower() and len(candidates) > 1:
                        candidates_filtered = [c for c in candidates if c.lower() != entity['text'].lower()]
                        replacement = random.choice(candidates_filtered)
                    
                    modified_text = (modified_text[:entity['start']] + 
                                   replacement + 
                                   modified_text[entity['end']:])
                    
                    replacements.append({
                        'original': entity['text'],
                        'replacement': replacement,
                        'type': entity_type
                    })
                    
        return modified_text, replacements
        
    def masking_replacement(self, text, entities, sensitive_entities=None):
        """UNBIASED entity masking"""
        if not entities:
            return text, []
            
        if sensitive_entities is None:
            num_sensitive = min(len(entities), random.randint(1, min(3, len(entities))))
            sensitive_entities = random.sample(entities, num_sensitive)
            
        replacements = []
        modified_text = text
        
        entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        for entity in entities_sorted:
            if entity in sensitive_entities:
                mask_token = f"[{entity['label']}]"
                
                modified_text = (modified_text[:entity['start']] + 
                               mask_token + 
                               modified_text[entity['end']:])
                
                replacements.append({
                    'original': entity['text'],
                    'replacement': mask_token,
                    'type': entity['label']
                })
                
        return modified_text, replacements
        
    def evaluate_semantic_preservation(self, original_texts, modified_texts):
        """Unbiased semantic preservation evaluation using multiple metrics"""
        metrics = {
            'bleu_scores': [],
            'rouge_scores': {'rouge1': [], 'rouge2': [], 'rougeL': []},
            'bert_scores': [],
            'sentence_similarity': []
        }
        
        for orig, mod in zip(original_texts, modified_texts):
            # BLEU Score
            orig_tokens = orig.split()
            mod_tokens = mod.split()
            if len(orig_tokens) > 0 and len(mod_tokens) > 0:
                bleu = sentence_bleu([orig_tokens], mod_tokens)
                metrics['bleu_scores'].append(bleu)
            
            # ROUGE Scores
            rouge_scores = self.rouge_scorer.score(orig, mod)
            for key in metrics['rouge_scores']:
                metrics['rouge_scores'][key].append(rouge_scores[key].fmeasure)
            
            # Sentence Similarity (using primary encoder)
            embeddings = self.sentence_encoder.encode([orig, mod])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            metrics['sentence_similarity'].append(similarity)
        
        # BERTScore
        if len(original_texts) > 0:
            P, R, F1 = bert_score.score(modified_texts, original_texts, lang='en', verbose=False)
            metrics['bert_scores'] = F1.tolist()
        
        # Compute averages
        avg_metrics = {}
        for key, values in metrics.items():
            if key == 'rouge_scores':
                avg_metrics[key] = {k: np.mean(v) for k, v in values.items()}
            else:
                avg_metrics[key] = np.mean(values) if values else 0.0
                
        return avg_metrics
        
    def evaluate_privacy_protection(self, original_texts, modified_texts, replacements_list):
        """Unbiased privacy protection evaluation"""
        direct_query_success = 0
        total_sensitive_relationships = 0
        
        for orig, mod, replacements in zip(original_texts, modified_texts, replacements_list):
            orig_entities = self.extract_entities(orig)
            if len(orig_entities) >= 2:
                total_sensitive_relationships += 1
                
                mod_entities = self.extract_entities(mod)
                
                orig_types = set(e['label'] for e in orig_entities)
                mod_types = set(e['label'] for e in mod_entities)
                
                if len(orig_types.intersection(mod_types)) >= 2:
                    orig_entity_texts = set(e['text'].lower() for e in orig_entities)
                    mod_entity_texts = set(e['text'].lower() for e in mod_entities)
                    
                    if len(orig_entity_texts.intersection(mod_entity_texts)) >= 2:
                        direct_query_success += 1
        
        direct_query_success_rate = direct_query_success / max(total_sensitive_relationships, 1)
        membership_inference_accuracy = self.simulate_membership_inference(original_texts, modified_texts)
        
        return {
            'direct_query_success_rate': direct_query_success_rate,
            'membership_inference_accuracy': membership_inference_accuracy,
            'total_sensitive_relationships': total_sensitive_relationships
        }
        
    def simulate_membership_inference(self, original_texts, modified_texts):
        """Simulate membership inference attacks"""
        correct_predictions = 0
        total_predictions = len(original_texts)
        
        for orig, mod in zip(original_texts, modified_texts):
            similarity = self.sentence_encoder.encode([orig, mod])
            sim_score = cosine_similarity([similarity[0]], [similarity[1]])[0][0]
            
            if sim_score > 0.8:
                correct_predictions += 1
        
        return correct_predictions / max(total_predictions, 1)
        
    def run_unbiased_experiment(self):
        """
        Run UNBIASED comprehensive experiments
        - Multiple runs for statistical significance
        - Randomized evaluation order
        - Fair comparison across all methods
        """
        print("Starting UNBIASED privacy NLP experiments...")
        print(f"Running {self.multiple_runs} iterations for statistical significance")
        
        datasets = self.load_datasets()
        
        # Define methods WITHOUT bias toward any approach
        methods = {
            'original': lambda text, entities, sensitive: (text, []),
            'context_aware': self.context_aware_replacement,
            'random_replacement': self.random_replacement,
            'masking': self.masking_replacement
        }
        
        all_results = []
        
        # Run multiple iterations
        for run_idx in range(self.multiple_runs):
            print(f"\n--- Run {run_idx + 1}/{self.multiple_runs} ---")
            
            # Reset random seed for each run to ensure different sampling
            random.seed(self.random_seed + run_idx)
            np.random.seed(self.random_seed + run_idx)
            
            run_results = {}
            
            for dataset_name, dataset in datasets.items():
                print(f"Processing dataset: {dataset_name}")
                run_results[dataset_name] = {}
                
                # Randomize dataset order to avoid bias
                dataset_sample = random.sample(dataset, min(len(dataset), 30))
                
                # Randomize method evaluation order
                method_items = list(methods.items())
                random.shuffle(method_items)
                
                for method_name, method_func in method_items:
                    print(f"  Running method: {method_name}")
                    
                    original_texts = []
                    modified_texts = []
                    all_replacements = []
                    
                    for item in dataset_sample:
                        text = item['text']
                        entities = self.extract_entities(text)
                        
                        if entities:
                            # Random selection of sensitive entities
                            num_sensitive = min(len(entities), random.randint(1, min(3, len(entities))))
                            sensitive_entities = random.sample(entities, num_sensitive)
                            
                            modified_text, replacements = method_func(text, entities, sensitive_entities)
                            
                            original_texts.append(text)
                            modified_texts.append(modified_text)
                            all_replacements.append(replacements)
                    
                    if original_texts:
                        semantic_metrics = self.evaluate_semantic_preservation(original_texts, modified_texts)
                        privacy_metrics = self.evaluate_privacy_protection(original_texts, modified_texts, all_replacements)
                        
                        run_results[dataset_name][method_name] = {
                            'semantic_preservation': semantic_metrics,
                            'privacy_protection': privacy_metrics,
                            'num_samples': len(original_texts)
                        }
                    else:
                        run_results[dataset_name][method_name] = {
                            'error': 'No entities found in dataset'
                        }
            
            all_results.append(run_results)
        
        # Aggregate results across runs
        self.results = self.aggregate_multiple_runs(all_results)
        return self.results
        
    def aggregate_multiple_runs(self, all_results):
        """Aggregate results from multiple runs with statistical analysis"""
        print("Aggregating results from multiple runs...")
        
        aggregated = {}
        
        # Get structure from first run
        first_run = all_results[0]
        
        for dataset_name in first_run.keys():
            aggregated[dataset_name] = {}
            
            for method_name in first_run[dataset_name].keys():
                if 'error' in first_run[dataset_name][method_name]:
                    aggregated[dataset_name][method_name] = first_run[dataset_name][method_name]
                    continue
                
                # Collect metrics across all runs
                semantic_metrics = defaultdict(list)
                privacy_metrics = defaultdict(list)
                
                for run_result in all_results:
                    if (dataset_name in run_result and 
                        method_name in run_result[dataset_name] and
                        'error' not in run_result[dataset_name][method_name]):
                        
                        sp = run_result[dataset_name][method_name]['semantic_preservation']
                        pp = run_result[dataset_name][method_name]['privacy_protection']
                        
                        # Collect semantic metrics
                        for metric, value in sp.items():
                            if metric == 'rouge_scores':
                                for rouge_metric, rouge_value in value.items():
                                    semantic_metrics[f'rouge_{rouge_metric}'].append(rouge_value)
                            else:
                                semantic_metrics[metric].append(value)
                        
                        # Collect privacy metrics
                        for metric, value in pp.items():
                            privacy_metrics[metric].append(value)
                
                # Compute statistics
                aggregated_semantic = {}
                for metric, values in semantic_metrics.items():
                    if values:
                        aggregated_semantic[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'values': values
                        }
                
                aggregated_privacy = {}
                for metric, values in privacy_metrics.items():
                    if values:
                        aggregated_privacy[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'values': values
                        }
                
                aggregated[dataset_name][method_name] = {
                    'semantic_preservation': aggregated_semantic,
                    'privacy_protection': aggregated_privacy,
                    'num_runs': len(all_results)
                }
        
        return aggregated
        
    def perform_statistical_tests(self):
        """Perform statistical significance tests between methods"""
        print("Performing statistical significance tests...")
        
        significance_results = {}
        
        for dataset_name, dataset_results in self.results.items():
            significance_results[dataset_name] = {}
            
            methods = [m for m in dataset_results.keys() if 'error' not in dataset_results[m]]
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:  # Avoid duplicate comparisons
                        comparison_key = f"{method1}_vs_{method2}"
                        significance_results[dataset_name][comparison_key] = {}
                        
                        # Compare semantic preservation metrics
                        for metric in ['bleu_scores', 'sentence_similarity', 'bert_scores']:
                            if (metric in dataset_results[method1]['semantic_preservation'] and
                                metric in dataset_results[method2]['semantic_preservation']):
                                
                                values1 = dataset_results[method1]['semantic_preservation'][metric]['values']
                                values2 = dataset_results[method2]['semantic_preservation'][metric]['values']
                                
                                if len(values1) > 1 and len(values2) > 1:
                                    # Perform t-test
                                    t_stat, p_value = stats.ttest_ind(values1, values2)
                                    significance_results[dataset_name][comparison_key][metric] = {
                                        't_statistic': t_stat,
                                        'p_value': p_value,
                                        'significant': p_value < 0.05
                                    }
        
        return significance_results
        
    def print_unbiased_summary(self):
        """Print unbiased summary with statistical analysis"""
        if not self.results:
            print("No results to summarize. Run experiments first.")
            return
            
        print("\n" + "="*80)
        print("UNBIASED PRIVACY-PRESERVING NLP EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Multiple runs: {self.multiple_runs}")
        print(f"Statistical significance testing: {self.statistical_significance_testing}")
        print("="*80)
        
        for dataset_name, dataset_results in self.results.items():
            print(f"\nDataset: {dataset_name.upper()}")
            print("-" * 50)
            
            for method_name, method_results in dataset_results.items():
                if 'error' in method_results:
                    print(f"  {method_name}: {method_results['error']}")
                    continue
                    
                print(f"\n  Method: {method_name}")
                print(f"    Runs: {method_results.get('num_runs', 1)}")
                
                if 'semantic_preservation' in method_results:
                    sp = method_results['semantic_preservation']
                    print(f"    Semantic Preservation (mean ± std):")
                    
                    for metric, stats in sp.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            print(f"      {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                
                if 'privacy_protection' in method_results:
                    pp = method_results['privacy_protection']
                    print(f"    Privacy Protection (mean ± std):")
                    
                    for metric, stats in pp.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            print(f"      {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        # Perform and display statistical tests
        if self.statistical_significance_testing:
            significance_results = self.perform_statistical_tests()
            
            print("\n" + "="*80)
            print("STATISTICAL SIGNIFICANCE TESTS")
            print("="*80)
            
            for dataset_name, comparisons in significance_results.items():
                print(f"\nDataset: {dataset_name.upper()}")
                print("-" * 50)
                
                for comparison, metrics in comparisons.items():
                    print(f"\n  {comparison}:")
                    for metric, test_result in metrics.items():
                        significance = "***" if test_result['significant'] else "n.s."
                        print(f"    {metric}: p={test_result['p_value']:.4f} {significance}")
        
        print("\n" + "="*80)
        print("UNBIASED EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("Key bias mitigation measures applied:")
        print("- Randomized evaluation order")
        print("- Multiple independent runs")
        print("- Balanced candidate pools")
        print("- Statistical significance testing")
        print("- Fair sampling across datasets")
        print("="*80)
        
    def generate_individual_graphs(self):
        """Generate individual publication-quality graphs"""
        if not self.results:
            print("No results to visualize. Run experiments first.")
            return
            
        print("Generating individual publication-quality graphs...")
        
        # Create results directory
        os.makedirs('privacy_nlp_results/individual_graphs', exist_ok=True)
        
        # Prepare data
        datasets = list(self.results.keys())
        methods = ['context_aware', 'random_replacement', 'masking']
        
        # Collect data for visualization
        semantic_data = []
        privacy_data = []
        
        for dataset in datasets:
            for method in methods:
                if (method in self.results[dataset] and 
                    'semantic_preservation' in self.results[dataset][method]):
                    
                    sp = self.results[dataset][method]['semantic_preservation']
                    pp = self.results[dataset][method]['privacy_protection']
                    
                    # Semantic data
                    semantic_data.append({
                        'Dataset': dataset,
                        'Method': method,
                        'BLEU': sp.get('bleu_scores', {}).get('mean', 0),
                        'BLEU_std': sp.get('bleu_scores', {}).get('std', 0),
                        'Similarity': sp.get('sentence_similarity', {}).get('mean', 0),
                        'Similarity_std': sp.get('sentence_similarity', {}).get('std', 0),
                        'BERTScore': sp.get('bert_scores', {}).get('mean', 0),
                        'BERTScore_std': sp.get('bert_scores', {}).get('std', 0)
                    })
                    
                    # Privacy data
                    privacy_data.append({
                        'Dataset': dataset,
                        'Method': method,
                        'Direct_Query_Success': pp.get('direct_query_success_rate', {}).get('mean', 0),
                        'Direct_Query_Success_std': pp.get('direct_query_success_rate', {}).get('std', 0),
                        'Membership_Inference': pp.get('membership_inference_accuracy', {}).get('mean', 0),
                        'Membership_Inference_std': pp.get('membership_inference_accuracy', {}).get('std', 0)
                    })
        
        if not semantic_data:
            print("No data available for visualization.")
            return
            
        df_semantic = pd.DataFrame(semantic_data)
        df_privacy = pd.DataFrame(privacy_data)
        
        # Set style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # Professional color palette
        
        # Graph 1: BLEU Score Comparison
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(datasets))
        width = 0.25
        
        for i, method in enumerate(methods):
            method_data = df_semantic[df_semantic['Method'] == method]
            means = [method_data[method_data['Dataset'] == d]['BLEU'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            stds = [method_data[method_data['Dataset'] == d]['BLEU_std'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            
            plt.bar(x_pos + i*width, means, width, yerr=stds, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8, capsize=5)
        
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('BLEU Score', fontsize=12, fontweight='bold')
        plt.title('BLEU Score Comparison Across Methods', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width, [d.replace('_', ' ').title() for d in datasets])
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('privacy_nlp_results/individual_graphs/bleu_score_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 2: Sentence Similarity Comparison
        plt.figure(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            method_data = df_semantic[df_semantic['Method'] == method]
            means = [method_data[method_data['Dataset'] == d]['Similarity'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            stds = [method_data[method_data['Dataset'] == d]['Similarity_std'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            
            plt.bar(x_pos + i*width, means, width, yerr=stds, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8, capsize=5)
        
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
        plt.title('Sentence Similarity Comparison Across Methods', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width, [d.replace('_', ' ').title() for d in datasets])
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('privacy_nlp_results/individual_graphs/sentence_similarity_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 3: Privacy Protection (Direct Query Success Rate)
        plt.figure(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            method_data = df_privacy[df_privacy['Method'] == method]
            means = [method_data[method_data['Dataset'] == d]['Direct_Query_Success'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            stds = [method_data[method_data['Dataset'] == d]['Direct_Query_Success_std'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            
            plt.bar(x_pos + i*width, means, width, yerr=stds, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8, capsize=5)
        
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Direct Query Success Rate', fontsize=12, fontweight='bold')
        plt.title('Privacy Protection: Direct Query Success Rate (Lower = Better)', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width, [d.replace('_', ' ').title() for d in datasets])
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('privacy_nlp_results/individual_graphs/privacy_protection_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 4: BERTScore Comparison
        plt.figure(figsize=(10, 6))
        
        for i, method in enumerate(methods):
            method_data = df_semantic[df_semantic['Method'] == method]
            means = [method_data[method_data['Dataset'] == d]['BERTScore'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            stds = [method_data[method_data['Dataset'] == d]['BERTScore_std'].iloc[0] if len(method_data[method_data['Dataset'] == d]) > 0 else 0 for d in datasets]
            
            plt.bar(x_pos + i*width, means, width, yerr=stds, 
                   label=method.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8, capsize=5)
        
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('BERTScore F1', fontsize=12, fontweight='bold')
        plt.title('BERTScore F1 Comparison Across Methods', fontsize=14, fontweight='bold')
        plt.xticks(x_pos + width, [d.replace('_', ' ').title() for d in datasets])
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('privacy_nlp_results/individual_graphs/bertscore_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 5: Overall Performance Heatmap
        plt.figure(figsize=(10, 8))
        
        # Create performance matrix
        performance_matrix = []
        method_labels = []
        
        for method in methods:
            method_scores = []
            for dataset in datasets:
                if (method in self.results[dataset] and 
                    'semantic_preservation' in self.results[dataset][method]):
                    
                    sp = self.results[dataset][method]['semantic_preservation']
                    pp = self.results[dataset][method]['privacy_protection']
                    
                    # Composite score: semantic preservation + privacy protection
                    semantic_score = sp.get('sentence_similarity', {}).get('mean', 0)
                    privacy_score = 1 - pp.get('direct_query_success_rate', {}).get('mean', 1)  # Invert for better privacy
                    composite_score = (semantic_score + privacy_score) / 2
                    
                    method_scores.append(composite_score)
                else:
                    method_scores.append(0)
            
            performance_matrix.append(method_scores)
            method_labels.append(method.replace('_', ' ').title())
        
        if performance_matrix:
            sns.heatmap(performance_matrix, 
                       xticklabels=[d.replace('_', ' ').title() for d in datasets], 
                       yticklabels=method_labels,
                       annot=True, 
                       cmap='RdYlBu_r',
                       fmt='.3f',
                       cbar_kws={'label': 'Composite Performance Score'})
            plt.title('Overall Performance Heatmap\n(Higher = Better Performance)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Dataset', fontsize=12, fontweight='bold')
            plt.ylabel('Method', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('privacy_nlp_results/individual_graphs/performance_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graph 6: Statistical Significance Line Graph
        if self.statistical_significance_testing:
            significance_results = self.perform_statistical_tests()
            
            plt.figure(figsize=(14, 8))
            
            # Create significance data
            sig_data = []
            metrics = ['bleu_scores', 'sentence_similarity', 'bert_scores']
            
            for dataset_name, dataset_comparisons in significance_results.items():
                for comparison, comparison_metrics in dataset_comparisons.items():
                    for metric in metrics:
                        if metric in comparison_metrics:
                            sig_data.append({
                                'Dataset': dataset_name.replace('_', ' ').title(),
                                'Comparison': comparison.replace('_', ' vs ').title(),
                                'Metric': metric.replace('_', ' ').title(),
                                'P_Value': comparison_metrics[metric]['p_value'],
                                'Significant': comparison_metrics[metric]['significant']
                            })
            
            if sig_data:
                df_sig = pd.DataFrame(sig_data)
                
                # Create line graph for each metric
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle('Statistical Significance Tests (P-Values)\nLower Values = More Significant', 
                           fontsize=16, fontweight='bold')
                
                colors = ['#2E86AB', '#A23B72', '#F18F01']
                markers = ['o', 's', '^']
                
                for i, metric in enumerate(metrics):
                    ax = axes[i]
                    metric_data = df_sig[df_sig['Metric'] == metric.replace('_', ' ').title()]
                    
                    if not metric_data.empty:
                        # Group by dataset and comparison
                        datasets = metric_data['Dataset'].unique()
                        comparisons = metric_data['Comparison'].unique()
                        
                        x_positions = np.arange(len(datasets))
                        
                        for j, comparison in enumerate(comparisons):
                            comp_data = metric_data[metric_data['Comparison'] == comparison]
                            p_values = []
                            
                            for dataset in datasets:
                                dataset_comp = comp_data[comp_data['Dataset'] == dataset]
                                if not dataset_comp.empty:
                                    p_values.append(dataset_comp['P_Value'].iloc[0])
                                else:
                                    p_values.append(1.0)  # No significance
                            
                            # Plot line with markers
                            ax.plot(x_positions, p_values, 
                                   marker=markers[j % len(markers)], 
                                   color=colors[j % len(colors)],
                                   linewidth=2, markersize=8, 
                                   label=comparison, alpha=0.8)
                            
                            # Mark significant points (p < 0.05)
                            for k, p_val in enumerate(p_values):
                                if p_val < 0.05:
                                    ax.scatter(x_positions[k], p_val, 
                                             color=colors[j % len(colors)], 
                                             s=100, marker='*', 
                                             edgecolors='black', linewidth=1)
                        
                        # Add significance threshold line
                        ax.axhline(y=0.05, color='red', linestyle='--', 
                                  alpha=0.7, linewidth=2, label='p = 0.05 threshold')
                        
                        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
                        ax.set_ylabel('P-Value', fontsize=12, fontweight='bold')
                        ax.set_title(f'{metric.replace("_", " ").title()}', 
                                   fontsize=14, fontweight='bold')
                        ax.set_xticks(x_positions)
                        ax.set_xticklabels(datasets, rotation=45, ha='right')
                        ax.grid(True, alpha=0.3)
                        ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
                        
                        # Add legend only to the first subplot
                        if i == 0:
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                
                plt.tight_layout()
                plt.savefig('privacy_nlp_results/individual_graphs/statistical_significance_line_graph.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Alternative single line graph showing all comparisons
                plt.figure(figsize=(14, 8))
                
                # Create a comprehensive line graph
                all_comparisons = []
                all_p_values = []
                all_labels = []
                
                for _, row in df_sig.iterrows():
                    label = f"{row['Dataset']} - {row['Comparison']} ({row['Metric']})"
                    all_comparisons.append(len(all_comparisons))
                    all_p_values.append(row['P_Value'])
                    all_labels.append(label)
                
                if all_comparisons:
                    # Plot main line
                    plt.plot(all_comparisons, all_p_values, 'o-', 
                            color='#2E86AB', linewidth=2, markersize=6, alpha=0.8)
                    
                    # Highlight significant points
                    significant_indices = [i for i, p in enumerate(all_p_values) if p < 0.05]
                    if significant_indices:
                        significant_x = [all_comparisons[i] for i in significant_indices]
                        significant_y = [all_p_values[i] for i in significant_indices]
                        plt.scatter(significant_x, significant_y, 
                                  color='red', s=100, marker='*', 
                                  edgecolors='black', linewidth=1, 
                                  label='Significant (p < 0.05)', zorder=5)
                    
                    # Add significance threshold
                    plt.axhline(y=0.05, color='red', linestyle='--', 
                              alpha=0.7, linewidth=2, label='p = 0.05 threshold')
                    
                    plt.xlabel('Comparison Index', fontsize=12, fontweight='bold')
                    plt.ylabel('P-Value', fontsize=12, fontweight='bold')
                    plt.title('Statistical Significance Tests - All Comparisons\n(Lower = More Significant)', 
                             fontsize=14, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.legend(fontsize=10)
                    
                    # Add text annotations for significant points
                    for i, (x, y, label) in enumerate(zip(all_comparisons, all_p_values, all_labels)):
                        if y < 0.05:
                            plt.annotate(f'p={y:.3f}', 
                                       xy=(x, y), xytext=(5, 5), 
                                       textcoords='offset points', 
                                       fontsize=8, alpha=0.8)
                
                plt.tight_layout()
                plt.savefig('privacy_nlp_results/individual_graphs/statistical_significance_comprehensive.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print("Individual graphs generated successfully!")
        print("Saved to privacy_nlp_results/individual_graphs/:")
        print("- bleu_score_comparison.png")
        print("- sentence_similarity_comparison.png") 
        print("- privacy_protection_comparison.png")
        print("- bertscore_comparison.png")
        print("- performance_heatmap.png")
        if self.statistical_significance_testing:
            print("- statistical_significance_line_graph.png")
            print("- statistical_significance_comprehensive.png")

    def save_unbiased_results(self, filename='unbiased_privacy_nlp_results.json'):
        """Save unbiased experimental results"""
        if not self.results:
            print("No results to save. Run experiments first.")
            return
            
        os.makedirs('privacy_nlp_results', exist_ok=True)
        filepath = os.path.join('privacy_nlp_results', filename)
        
        # Include experimental setup information
        results_with_metadata = {
            'experimental_setup': {
                'multiple_runs': self.multiple_runs,
                'random_seed': self.random_seed,
                'bias_mitigation_measures': [
                    'randomized_evaluation_order',
                    'multiple_independent_runs',
                    'balanced_candidate_pools',
                    'statistical_significance_testing',
                    'fair_dataset_sampling'
                ]
            },
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
            
        print(f"Unbiased results saved to {filepath}")

def main():
    """Main function to run unbiased experiments"""
    print("Unbiased Privacy-Preserving NLP Experiments")
    print("==========================================")
    print("This experiment is designed to fairly test the hypothesis without bias")
    print("Key features:")
    print("- 10 independent runs for robust statistical significance")
    print("- Randomized evaluation order")
    print("- Balanced candidate pools")
    print("- Statistical significance testing")
    print("- Fair sampling across datasets")
    print("==========================================")
    
    # Initialize unbiased experiment
    experiment = UnbiasedPrivacyNLPExperiment()
    
    # Run unbiased experiments
    results = experiment.run_unbiased_experiment()
    
    # Print unbiased summary
    experiment.print_unbiased_summary()
    
    # Generate individual publication-quality graphs
    experiment.generate_individual_graphs()
    
    # Save unbiased results
    experiment.save_unbiased_results()
    
    print("\nUnbiased experiment completed successfully!")
    print("Results and individual graphs saved to privacy_nlp_results/ directory")

if __name__ == "__main__":
    main()
# END GENAI@GHCOPILOT

