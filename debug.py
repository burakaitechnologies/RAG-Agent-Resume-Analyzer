"""
Debug script for RAG Agent - helps identify specific issues
Run this script to diagnose problems with your setup
"""

import os
import sys
from dotenv import load_dotenv

def test_environment_variables():
    """Test all required environment variables"""
    print("🔍 Testing Environment Variables...")
    load_dotenv()
    
    required_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY"), 
        "INDEX_NAME": os.environ.get("INDEX_NAME"),
        "LANGCHAIN_API_KEY": os.environ.get("LANGCHAIN_API_KEY")
    }
    
    for var, value in required_vars.items():
        if value:
            print(f"   ✅ {var}: {'*' * 10}...{value[-4:] if len(value) > 4 else '*' * len(value)}")
        else:
            print(f"   ❌ {var}: Not set")
    
    return all(required_vars.values())

def test_pinecone_connection():
    """Test Pinecone connection and index status"""
    print("\n🔍 Testing Pinecone Connection...")
    
    try:
        from pinecone import Pinecone
        load_dotenv()
        
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        print("   ✅ Pinecone connection successful")
        
        # List indexes
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        print(f"   📋 Available indexes: {index_names}")
        
        target_index = os.environ.get("INDEX_NAME")
        if target_index in index_names:
            print(f"   ✅ Target index '{target_index}' exists")
            
            # Get index stats
            index = pc.Index(target_index)
            stats = index.describe_index_stats()
            print(f"   📊 Index stats: {stats.total_vector_count} vectors")
            
        else:
            print(f"   ⚠️  Target index '{target_index}' does not exist")
            print("   💡 The app will try to create it automatically")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pinecone connection failed: {str(e)}")
        return False

def test_openai_connection():
    """Test OpenAI connection"""
    print("\n🔍 Testing OpenAI Connection...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        load_dotenv()
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        
        # Test embedding
        test_text = "This is a test document for embedding."
        result = embeddings.embed_query(test_text)
        
        print(f"   ✅ OpenAI embeddings working (dimension: {len(result)})")
        return True
        
    except Exception as e:
        print(f"   ❌ OpenAI connection failed: {str(e)}")
        return False

def test_document_loading():
    """Test document loading from sample files"""
    print("\n🔍 Testing Document Loading...")
    
    sample_path = "sample_documents"
    if os.path.exists(sample_path):
        files = os.listdir(sample_path)
        print(f"   📁 Sample documents folder found with {len(files)} files:")
        
        for file in files:
            print(f"      - {file}")
        
        # Test loading one file
        test_file = os.path.join(sample_path, "resume_john_smith.txt")
        if os.path.exists(test_file):
            try:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(test_file, encoding='utf-8')
                docs = loader.load()
                print(f"   ✅ Successfully loaded test file: {len(docs)} documents")
                print(f"   📄 Sample content: {docs[0].page_content[:100]}...")
                return True
            except Exception as e:
                print(f"   ❌ Error loading test file: {str(e)}")
                return False
        else:
            print(f"   ⚠️  Test file not found: {test_file}")
            return False
    else:
        print(f"   ❌ Sample documents folder not found: {sample_path}")
        return False

def test_full_pipeline():
    """Test the complete pipeline with a small document"""
    print("\n🔍 Testing Full Pipeline...")
    
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone
        
        load_dotenv()
        
        # Create a test document
        test_content = """
        This is a test resume for John Doe.
        
        Skills: Python, JavaScript, React
        Experience: 5 years in software development
        Education: Computer Science degree
        """
        
        # Create temporary file
        with open("temp_test.txt", "w") as f:
            f.write(test_content)
        
        # Load document
        loader = TextLoader("temp_test.txt", encoding='utf-8')
        docs = loader.load()
        print(f"   ✅ Document loaded: {len(docs)} documents")
        
        # Split document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        splits = text_splitter.split_documents(docs)
        print(f"   ✅ Document split: {len(splits)} chunks")
        
        # Test embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
        test_embedding = embeddings.embed_query("test")
        print(f"   ✅ Embeddings working: dimension {len(test_embedding)}")
        
        # Clean up
        os.remove("temp_test.txt")
        
        print("   ✅ Full pipeline test successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {str(e)}")
        # Clean up on error
        if os.path.exists("temp_test.txt"):
            os.remove("temp_test.txt")
        return False

def main():
    """Run all diagnostic tests"""
    print("=" * 60)
    print("🏥 HR RAG Agent - Diagnostic Script")
    print("=" * 60)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("OpenAI Connection", test_openai_connection),
        ("Pinecone Connection", test_pinecone_connection),
        ("Document Loading", test_document_loading),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 {test_name} crashed: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup should work.")
        print("\n💡 If you're still getting errors, try:")
        print("   1. Use the sample_documents directory path")
        print("   2. Check the Flask console for detailed error messages")
        print("   3. Make sure your Pinecone free tier has space")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n🔧 Common solutions:")
        print("   1. Check your .env file has all required API keys")
        print("   2. Verify your Pinecone API key and index name")
        print("   3. Ensure you have an active OpenAI account with credits")
        print("   4. Try creating a new Pinecone index if needed")

if __name__ == "__main__":
    main()
