import os
import re
import json

try:
    from langchain_community.llms import Ollama
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    import requests
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n🔧 Please install required packages:")
    print("pip install langchain langchain-community langchain-text-splitters chromadb ollama requests")
    exit(1)
from dotenv import load_dotenv
from backend.file_manager import FileManager

load_dotenv()

class RAGSystem:
    def __init__(self, md_file_path=None):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.file_manager = FileManager()
        self.vector_store = None

        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url
        )
        self.classifier_llm = Ollama(
            model=self.default_model,
            temperature=float(os.getenv("OLLAMA_CLASSIFIER_TEMPERATURE", "0.1")),
            base_url=self.ollama_base_url
        )
        self.answer_llm = None
        self.answer_model_name = None
        
        # Fast pattern matching for OBVIOUS cases only
        self.simple_patterns = {
            'greeting': [
                r'^(hi|hello|hey|good morning|good afternoon|good evening)[\s!]*$',
                r'^how are you[\s\?!]*$',
                r'^what\'s up[\s\?!]*$',
            ],
            'farewell': [
                r'^(bye|goodbye|see you|take care|bye bye)[\s!]*$',
            ],
            'thanks': [
                r'^(thanks|thank you|thx|appreciate it)[\s!]*$',
            ],
            'acknowledgment': [  # NEW: For phrases like "okay i understand"
                r'^(okay|ok|alright|i understand|got it|understood)[\s!]*$',
                r'^(yes|yeah|sure)[\s!]*$',
            ]
        }
        
        # Check server status pehle
        if not self.check_server_status():
            print("⚠️ WARNING: Cannot connect to remote Ollama server!")
            print(f"   Make sure {self.ollama_base_url} is accessible")
        else:
            print("✅ Successfully connected to remote Ollama server!")
            
            # Available models check karo
            models = self.get_available_models()
            if models:
                print(f"📋 Available models on remote server: {', '.join(models)}")
                if self.default_model not in models and f"{self.default_model}:latest" not in models:
                    print(f"⚠️ Warning: {self.default_model} may not be available on server")
        
        # Initialize vector store
        self.load_documents()
    
    def check_server_status(self):
        """Remote Ollama server accessible hai ya nahi check karo"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Server connection error: {e}")
            return False
    
    def get_available_models(self):
        """Remote server pe kon se models available hain"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            print(f"⚠️ Error getting models: {e}")
            return []
    
    def load_documents(self):
        """Load existing vector store if available, else do nothing"""
        try:
            if os.path.exists("./chroma_db"):
                # Load existing vectorstore (no embedding creation)
                print("✅ Loading existing vectorstore from chroma_db...")
                self.vector_store = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings
                )
                print("✅ Vectorstore loaded from disk")
            else:
                print("⚠️ No existing vectorstore found (will create on upload)")
                self.vector_store = None
        except Exception as e:
            print(f"❌ Error loading vectorstore: {e}")
            self.vector_store = None
    
    def reload_documents(self):
        """Recreate vectorstore from current document (for upload/delete)"""
        try:
            # FileManager se current document ka path lo
            doc_path = self.file_manager.get_document_path()
        
            if not doc_path:
                print("❌ No document to reload")
                self.vector_store = None
                return
        
            # Purana chroma_db delete karo
            if os.path.exists("./chroma_db"):
                import shutil
                shutil.rmtree("./chroma_db")
                print("🗑️ Cleared old vectorstore")
        
            # Naya content load karo aur embeddings create karo
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
            # IMPROVED CHUNKING: Headers ko zyada prioritize karo
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,  # Thoda chhota karo taake sections better separate ho
                chunk_overlap=100,  # Overlap kam karo
                length_function=len,
                separators=["\n### ", "\n## ", "\n#### ", "\n\n", "\n", ". ", " "]  # Headers pehle
            )
        
            chunks = text_splitter.split_text(content)
        
            print(f"📦 Creating new vector embeddings for {len(chunks)} chunks...")
            self.vector_store = Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
        
            print(f"✅ Vectorstore recreated from {doc_path}")
        
        except Exception as e:
            print(f"❌ Error reloading documents: {e}")
            self.vector_store = None
    
    def quick_pattern_check(self, question):
        """
        FAST pattern matching for OBVIOUS simple cases
        Returns: (is_simple, response_type) or (False, None)
        """
        text = question.lower().strip()
        
        # Only check if query is SHORT (less than 6 words)
        if len(text.split()) > 5:
            return False, None
        
        # Check simple patterns
        for pattern_type, patterns in self.simple_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    return True, pattern_type
        
        return False, None
    
    def fast_intent_classify(self, question, conversation_history=None):
        text = question.lower().strip()
        if any(word in text for word in ["summary", "overview", "high level", "main points"]):
            return "DOCUMENT_SUMMARY"
        if conversation_history and len(conversation_history) >= 2:
            if any(phrase in text for phrase in ["more detail", "more details", "elaborate", "explain more", "tell me more", "what about", "clarify"]):
                return "FOLLOWUP"
        if not conversation_history:
            tokens = text.split()
            if any(token in tokens for token in ["what", "how", "when", "where", "why", "does", "do", "can", "is", "are"]):
                return "DOCUMENT_QUERY"
        return None
    
    def classify_with_llm(self, question, conversation_history=None):
        """
        LLM classification for COMPLEX or AMBIGUOUS cases
        """
        # Build conversation context
        conv_context = ""
        if conversation_history and len(conversation_history) >= 2:
            recent = conversation_history[-4:]
            for msg in recent:
                role = "User" if msg['role'] == 'user' else "Assistant"
                conv_context += f"{role}: {msg['content'][:200]}\n"
        
        classification_prompt = f"""Classify this user query into ONE category:

Categories:
- DOCUMENT_SUMMARY: asking for document overview/summary
- FOLLOWUP: asking for more details/clarification about previous answer
- DOCUMENT_QUERY: specific question about document content
- ACKNOWLEDGMENT: simple acknowledgment like "okay" or "i understand"
- OFFTOPIC: not related to documents

{"Previous Conversation:" if conv_context else "No previous conversation."}
{conv_context}

User Question: "{question}"

IMPORTANT: If the query is a simple acknowledgment or greeting, classify as ACKNOWLEDGMENT. Respond with ONLY the category name (one word):"""

        try:
            response = self.classifier_llm.invoke(classification_prompt)
            response = response.strip().upper()
            
            # Extract category
            for category in ['DOCUMENT_SUMMARY', 'FOLLOWUP', 'DOCUMENT_QUERY', 'ACKNOWLEDGMENT', 'OFFTOPIC']:
                if category in response:
                    return category
            
            return 'DOCUMENT_QUERY'
            
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return 'DOCUMENT_QUERY'
    
    def handle_simple_response(self, response_type):
        """Handle simple pattern-matched responses with more personality"""
        responses = {
            'greeting': "Hello! I'm your document assistant. I'm here to help with questions about the uploaded documents. What would you like to know?",
            'farewell': "Goodbye! Feel free to come back anytime for more help with the documents.",
            'thanks': "You're very welcome! Let me know if you have any other questions about the documents.",
            'acknowledgment': "Great! If you have more questions about the documents or need clarification, just ask."
        }
        return responses.get(response_type, "How can I help you with the documents?")
    
    def get_document_summary(self, model_name=None):
        """Get comprehensive document summary"""
        try:
            if not self.vector_store:
                return "Document not loaded."
            
            all_docs = self.vector_store.get()
            if not all_docs or 'documents' not in all_docs or not all_docs['documents']:
                return "Unable to generate summary."
            
            intro_text = "\n\n".join(all_docs['documents'][:10])

            llm = Ollama(
                model=model_name or self.default_model,
                temperature=0.3,
                base_url=self.ollama_base_url
            )
            
            prompt = f"""Based on the document content below, provide a comprehensive summary:

1. Main topic/purpose
2. Key sections covered
3. Important highlights

Document Content:
{intro_text[:3000]}

Summary:"""
            
            summary = llm.invoke(prompt)
            return summary
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def build_conversation_context(self, conversation_history):
        """Build context from recent conversation"""
        if not conversation_history or len(conversation_history) < 2:
            return ""
        
        recent = conversation_history[-4:]
        context_parts = []
        
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content'][:200]}")
        
        return "\n".join(context_parts)
    
    def expand_query_with_context(self, question, conversation_history):
        """
        🔥 KEY FIX: Expand pronoun-heavy questions using conversation context
        Example: "how many total these leaves?" -> "how many total sick leave and casual leave days?"
        """
        if not conversation_history or len(conversation_history) < 2:
            return question
        
        # Get last 4 messages (2 exchanges)
        recent = conversation_history[-4:]
        context = "\n".join([f"{msg['role']}: {msg['content'][:150]}" for msg in recent])
        
        expansion_prompt = f"""Rewrite the user's question to be self-contained by replacing pronouns and vague references with specific terms from the conversation context.

Previous Conversation:
{context}

Current Question: "{question}"

Instructions:
- Replace "these", "those", "it", "that" with specific nouns from conversation
- Replace "total", "both" with actual items mentioned
- Keep the question SHORT and clear (under 15 words)
- Output ONLY the rewritten question, nothing else

Rewritten Question:"""

        try:
            expanded = self.classifier_llm.invoke(expansion_prompt)
            expanded = expanded.strip().split('\n')[0]  # Take first line only
            
            # Clean up response
            expanded = expanded.replace('"', '').replace('Rewrite:', '').replace('Question:', '').replace('Rewritten:', '').strip()
            
            # Validate expansion
            if len(expanded.split()) > 20 or len(expanded) < 5:
                print(f"⚠️ Expansion too long/short, using original")
                return question
            
            print(f"🔍 Query Expanded: '{question}' → '{expanded}'")
            return expanded
            
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return question
    
    def query(self, question, model_name=None, conversation_history=None):
        """
        HYBRID APPROACH with QUERY EXPANSION:
        1. Fast patterns for simple cases
        2. LLM classification for complex ones
        3. Query expansion for pronoun-heavy questions
        4. Context-aware retrieval
        """
        try:
            # NEW: Agar vectorstore nahi hai, to reload karo (load existing ya create new)
            if not self.vector_store:
                print("Vectorstore not ready, reloading...")
                self.reload_documents()

            if not self.vector_store:
                return "Document database not ready. Please check permissions or upload a document."
        
            is_simple, response_type = self.quick_pattern_check(question)
            if is_simple:
                print(f"⚡ Fast match: {response_type}")
                return self.handle_simple_response(response_type)
            
            intent = self.fast_intent_classify(question, conversation_history)
            if not intent:
                intent = self.classify_with_llm(question, conversation_history)
            print(f"🧠 LLM classified as: {intent}")
            
            # STEP 3: Handle based on classification
            if intent == 'OFFTOPIC':
                return "I'm here to help with questions about the documents. Could you ask something related to the document content?"
            
            elif intent == 'DOCUMENT_SUMMARY':
                print("📄 Generating document summary...\n")
                return self.get_document_summary(model_name)
            
            elif intent == 'ACKNOWLEDGMENT':  # NEW: Handle simple acknowledgments
                return self.handle_simple_response('acknowledgment')
            
            # STEP 4: RAG pipeline for DOCUMENT_QUERY and FOLLOWUP
            # KEY FIX: Expand query with conversation context for better retrieval
            search_query = question
            if intent in ['FOLLOWUP', 'DOCUMENT_QUERY'] and conversation_history and len(conversation_history) >= 2:
                lowered = f" {question.lower()} "
                pronouns = [" it ", " this ", " that ", " these ", " those ", " them ", " both ", " all of them "]
                if any(p in lowered for p in pronouns):
                    search_query = self.expand_query_with_context(question, conversation_history)
            
            history_context = self.build_conversation_context(conversation_history)
            
            model_to_use = model_name or self.default_model
            if self.answer_llm is None or self.answer_model_name != model_to_use:
                self.answer_llm = Ollama(
                    model=model_to_use,
                    temperature=0.1,
                    base_url=self.ollama_base_url
                )
                self.answer_model_name = model_to_use
            llm = self.answer_llm
            
            # Choose prompt based on intent (UPDATED TEMPLATES - EVEN STRICTER)
            if intent == 'FOLLOWUP' and history_context:
                template = """You are a document assistant. User wants MORE DETAILS about previous discussion.

Previous Conversation:
{history}

Relevant Document Information:
{context}

Question: {question}

INSTRUCTIONS:
1. User wants clarification/elaboration on previous topic
2. Look at conversation to understand WHICH topic
3. Provide DETAILED explanation with specific numbers and facts FROM THE DOCUMENT ONLY
4. Use simple language
5. Be thorough but focused
6. If information is not in the provided document context, say: "This information is not found in the document."
7. Do NOT add external knowledge or make assumptions.

Detailed Answer:"""
            
            elif history_context:
                template = """You are a document assistant with conversation awareness.

Previous Conversation:
{history}

Document Information:
{context}

Question: {question}

INSTRUCTIONS:
1. Consider conversation context
2. Answer based ONLY on the provided document information
3. Be clear, specific, and precise with numbers/facts
4. Do not add external knowledge or assumptions
5. If asking about totals, calculate them from document data
6. If information is not in the document context, say: "This information is not found in the document."
7. Do NOT invent or hallucinate information.

Answer:"""
            
            else:
                template = """You are a document assistant.

Document Information:
{context}

Question: {question}

INSTRUCTIONS:
1. Answer ONLY from the provided document information
2. Be clear, specific, and precise
3. Do not add external knowledge, opinions, or unrelated content
4. If information is not available in the document, say: "This information is not found in the document."
5. Do NOT make up or assume information.

Answer:"""
            
            prompt = PromptTemplate.from_template(template)
            
            # More chunks for better context (INCREASED)
            k_value = 6 if intent == 'FOLLOWUP' else 4  # Increased further
            
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_value}
            )
            
            # IMPROVED format_docs with BETTER relevance filter
            def format_docs(docs):
                # Filter short chunks and prioritize those containing query keywords
                query_keywords = set(question.lower().split())
                relevant_docs = []
                for d in docs:
                    content = d.page_content.lower()
                    if len(content.strip()) > 50:  # Filter short
                        # Boost if keywords match
                        if any(kw in content for kw in query_keywords):
                            relevant_docs.append(d)
                        elif len(relevant_docs) < k_value // 2:  # Fallback to some non-matching
                            relevant_docs.append(d)
                
                # Log retrieved chunks for debugging
                print(f"🔍 Retrieved {len(relevant_docs)} relevant chunks for query")
                
                return "\n\n---\n\n".join([d.page_content for d in relevant_docs[:k_value]])
            
            # 🔥 KEY FIX: Use expanded query for retrieval, original for generation
            if history_context:
                chain = (
                    {
                        "context": lambda _: format_docs(retriever.invoke(search_query)),  # Expanded query
                        "question": RunnablePassthrough(),  # Original question
                        "history": lambda _: history_context
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            else:
                chain = (
                    {
                        "context": lambda _: format_docs(retriever.invoke(search_query)),  # Expanded query
                        "question": RunnablePassthrough()  # Original question
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            
            response = chain.invoke(question).strip()  # Pass original question to LLM
            
            # POST-PROCESSING: Add note if info not found
            if "not found" in response.lower() or "not available" in response.lower():
                response += " Please check the document or ask about a different topic."
            
            return response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
