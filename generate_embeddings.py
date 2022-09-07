import argparse
import json
import os

from models import HFmodel, miniLMSPECTER, scibertSPECTER

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="path to the pretrained model")
parser.add_argument(
    "--paper_data", help="path to the paper metadata json file", required=True
)
parser.add_argument(
    "--output",
    help="path to the file where the embeddings will be written to",
    required=True,
)
parser.add_argument("--batch_size", help="batch_size", type=int, default=2)
parser.add_argument(
    "--pooling",
    default=None,
    type=str,
    help="pooling mechanism during training (mean, cls, pretrain).",
)
parser.add_argument(
    "--sep",
    default=None,
    type=str,
    help="set to 'tokenizer' to use the tokenizer sep token, otherwise use a space instead",
)
args = parser.parse_args()

if os.path.exists(args.model):
    try:
        model = miniLMSPECTER(model_path=args.model, pooling=args.pooling, sep=args.sep)
    except:
        model = scibertSPECTER(
            model_path=args.model, pooling=args.pooling, sep=args.sep
        )
else:
    model = HFmodel(model_path=args.model, pooling=args.pooling, sep=args.sep)

d = None
with open(args.paper_data) as file:  # "../scidocs/data/paper_metadata_mag_mesh.json"
    d = json.load(file)

print("Extracting papers...", end=" ")
papers = []
for paper_id, paper_metadata in d.items():
    paper = {}
    paper["paper_id"] = paper_id
    paper["title"] = paper_metadata["title"]
    paper["text"] = paper_metadata["abstract"]  # rename abstract key into "text"
    papers.append(paper)

print("Done")
print(f"Extracted {len(papers)} papers")

print("Computing embeddings...")
embeddings = model.encode_corpus(corpus=papers, batch_size=args.batch_size)
print("Finished computing embeddings")

print("Writing to file...")
with open(args.output, "a") as file:  # "mag_mesh_embed.jsonl"
    for j in range(embeddings.shape[0]):
        paper_data = papers[j]
        paper_data["abstract"] = paper_data.pop(
            "text"
        )  # scidocs embeddings need should have an abstract
        paper_data["embedding"] = embeddings[j].tolist()
        json.dump(paper_data, file)
        file.write("\n")
print("Finished writing to file")
