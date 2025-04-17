from functools import wraps
from unstructured.partition.pdf import partition_pdf
from base64 import b64decode
from PIL import Image
from io import BytesIO


def with_chunks(func):
    """Decorator to handle PDF chunk creation."""
    @wraps(func)
    def wrapper(path, *args, **kwargs):
        chunks = partition_pdf(
            filename=path,
            infer_table_structure=True,
            strategy="fast",
            # extract_image_block_types=["Image"],
            # extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        return func(chunks, *args, **kwargs)
    return wrapper



@with_chunks
def extract_elements(chunks):
    texts, tables, images = [], [], []

    for chunk in chunks:
        chunk_type = str(type(chunk))

        if "CompositeElement" in chunk_type:
            texts.append(chunk)
            # for el in chunk.metadata.orig_elements:
            #     if "Image" in str(type(el)):
            #         img_b64 = getattr(el.metadata, "image_base64", None)
            #         if img_b64:
            #             try:
            #                 image = Image.open(BytesIO(b64decode(img_b64)))
            #                 images.append(image)
            #             except Exception as e:
            #                 print(f"⚠️ Skipping invalid image: {e}")

        elif "Table" in chunk_type:
            tables.append(chunk)

    return {
        "texts": texts,
        "tables": tables,
        "images": images,
    }
