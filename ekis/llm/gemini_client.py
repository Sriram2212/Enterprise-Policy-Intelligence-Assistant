"""Gemini LLM client module."""

from typing import Optional


class GeminiClient:
    """Client for interacting with Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        """Initialize Gemini client.
        
        Args:
            api_key: Google API key (if None, uses GOOGLE_API_KEY env var)
            model_name: Model name to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Gemini client."""
        try:
            import google.generativeai as genai
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
            
            self.client = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("Please install google-generativeai: pip install google-generativeai")
        except Exception as e:
            raise Exception(f"Error initializing Gemini client: {e}")

    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        if self.client is None:
            raise ValueError("Client not initialized")
        
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating response: {e}")

    def generate_stream(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7):
        """Generate response with streaming.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Yields:
            Response chunks
        """
        if self.client is None:
            raise ValueError("Client not initialized")
        
        try:
            response = self.client.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise Exception(f"Error generating streaming response: {e}")
