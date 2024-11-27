import sqlite3
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
from typing import List, Dict, Any

class RAGActiveLearner:
    def __init__(self, 
                 database_path: str = 'feedback.db', 
                 model_path: str = None, 
                 log_dir: str = 'active_learning_logs'):
        """
        Initialize the Active Learning system for RAG optimization.
        
        Args:
            database_path (str): Path to the SQLite feedback database
            model_path (str): Path to the base model to fine-tune
            log_dir (str): Directory for logging training progress
        """
        # Use the LLM model from the environment configuration
        self.base_model_name = os.getenv('llm_model', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
        
        # Database connection
        self.conn = sqlite3.connect(database_path)
        self.log_dir = log_dir
        
        # Model and tokenizer setup
        self.model_path = model_path or self.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def load_feedback_data(self, min_rating: int = 4) -> pd.DataFrame:
        """
        Load feedback data from the database, filtering by minimum rating.
        
        Args:
            min_rating (int): Minimum rating to consider for fine-tuning
        
        Returns:
            pd.DataFrame: Filtered feedback data
        """
        query = f"SELECT query, answer, rating FROM Feedback WHERE rating >= {min_rating}"
        df = pd.read_sql_query(query, self.conn)
        
        # Create training format
        df['training_text'] = df.apply(
            lambda row: f"Query: {row['query']}\nAnswer: {row['answer']}", 
            axis=1
        )
        
        return df
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dict[str, Dataset]:
        """
        Prepare dataset for fine-tuning, splitting into train and validation sets.
        
        Args:
            df (pd.DataFrame): Input dataframe with training data
        
        Returns:
            Dict containing train and validation datasets
        """
        # Tokenize the data
        def tokenize_function(examples):
            return self.tokenizer(
                examples['training_text'], 
                padding='max_length', 
                truncation=True, 
                max_length=512
            )
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[['training_text']])
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Split into train and validation
        train_dataset, val_dataset = tokenized_dataset.train_test_split(test_size=0.2).values()
        
        return {
            'train': train_dataset,
            'validation': val_dataset
        }
    
    def fine_tune(self, datasets: Dict[str, Dataset]):
        """
        Fine-tune the model using the prepared datasets.
        
        Args:
            datasets (Dict[str, Dataset]): Train and validation datasets
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.log_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(self.log_dir, 'logs'),
            logging_steps=10,
            evaluation_strategy="epoch"
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation']
        )
        
        # Train the model
        trainer.train()
        
        # Save the fine-tuned model
        fine_tuned_model_path = os.path.join(self.log_dir, 'fine_tuned_model')
        trainer.save_model(fine_tuned_model_path)
        print(f"Fine-tuned model saved to {fine_tuned_model_path}")
    
    def analyze_feedback_trends(self) -> Dict[str, Any]:
        """
        Analyze feedback trends and generate insights.
        
        Returns:
            Dict containing analysis insights
        """
        df = self.load_feedback_data()
        
        return {
            'total_feedback_count': len(df),
            'average_rating': df['rating'].mean(),
            'rating_distribution': df['rating'].value_counts(normalize=True).to_dict(),
            'most_problematic_queries': self._find_problematic_queries(df)
        }
    
    def _find_problematic_queries(self, df: pd.DataFrame, threshold: int = 2) -> List[Dict]:
        """
        Find queries with low ratings for further investigation.
        
        Args:
            df (pd.DataFrame): Input dataframe
            threshold (int): Rating threshold for problematic queries
        
        Returns:
            List of problematic query details
        """
        problematic = df[df['rating'] <= threshold]
        return problematic[['query', 'answer', 'rating']].to_dict('records')
    
    def run_active_learning_cycle(self):
        """
        Execute a complete active learning cycle:
        1. Load and prepare feedback data
        2. Fine-tune the model
        3. Generate insights
        """
        # Load feedback data
        df = self.load_feedback_data()
        
        if len(df) > 0:
            # Prepare dataset
            datasets = self.prepare_dataset(df)
            
            # Fine-tune model
            self.fine_tune(datasets)
            
            # Analyze feedback trends
            insights = self.analyze_feedback_trends()
            print("Active Learning Cycle Insights:", insights)
        else:
            print("No sufficient feedback data for active learning.")
    
    def __del__(self):
        """
        Close database connection when object is deleted.
        """
        if hasattr(self, 'conn'):
            self.conn.close()

# Example usage
if __name__ == "__main__":
    learner = RAGActiveLearner()
    learner.run_active_learning_cycle()