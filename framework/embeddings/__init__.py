try:
    from embeddings.embedding_esm import EsmEmbedding
except ImportError:
    print("You are missing some of the libraries for ESM")
try:
    from embeddings.embedding_protein_trans import ProteinTransEmbedding
except ImportError:
    print("You are missing some of the libraries for ProteinTrans")
try:
    from embeddings.embedding_protein_vec import ProteinVecEmbedding
except ImportError:
    print("You are missing some of the libraries for ProteinVec")
try:
    from embeddings.embedding_esm3 import Esm3Embedding
except ImportError:
    print("You are missing some of the libraries for ESM3")
