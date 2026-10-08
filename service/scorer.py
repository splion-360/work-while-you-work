import hashlib
import json
import math
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


GROUP_WEIGHTS = {
    "required": 0.35,
    "core": 0.25,
    "responsibilities": 0.20,
    "preferred": 0.10,
    "domain": 0.10,
}
CHUNK_TEXT_LIMIT = 1500
MAX_CRITERIA = 64
MAX_EVIDENCE = 96

SKILLS = {
    "AI safety": ("ai safety", "model safety"),
    "Adversarial testing": ("adversarial testing", "red teaming", "red-teaming"),
    "Airbyte": ("airbyte",),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure",),
    "C++": ("c++",),
    "CUDA": ("cuda",),
    "Datadog": ("datadog",),
    "Deep learning": ("deep learning",),
    "Distributed systems": ("distributed systems", "distributed computing"),
    "Docker": ("docker",),
    "FastAPI": ("fastapi",),
    "FSDP": ("fsdp", "fully sharded data parallel"),
    "GCP": ("gcp", "google cloud platform"),
    "Hugging Face": ("hugging face",),
    "Kubernetes": ("kubernetes", "k8s"),
    "LLMs": ("llm", "llms", "large language model", "large language models"),
    "Model monitoring": ("model monitoring", "continuous monitoring"),
    "ONNX": ("onnx",),
    "OpenCV": ("opencv",),
    "Parallel computing": ("parallel computing", "parallel training"),
    "PostgreSQL": ("postgresql", "postgres"),
    "PyTorch": ("pytorch",),
    "Python": ("python",),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "Redis": ("redis",),
    "Robustness testing": ("robustness testing", "robust inference", "model robustness"),
    "ROS2": ("ros2", "robot operating system 2"),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "Snowflake": ("snowflake",),
    "SQL": ("sql",),
    "Synthetic data": ("synthetic data",),
    "Temporal": ("temporal",),
    "Transformers": ("transformer", "transformers"),
    "TypeScript": ("typescript",),
    "Weights & Biases": ("weights & biases", "wandb", "w&b"),
}

SECTION_HEADINGS = {
    "the role": "domain", "role": "domain", "role overview": "domain",
    "about the job": "core",
    "what you will do": "responsibilities", "what you'll do": "responsibilities",
    "responsibilities": "responsibilities", "key responsibilities": "responsibilities",
    "who you are": "core", "experience": "core", "qualifications": "core",
    "core qualifications": "core", "core technical skills": "core", "skills": "core",
    "requirements": "required", "minimum qualifications": "required",
    "required qualifications": "required", "education": "required",
    "preferred qualifications": "preferred", "preferred experience": "preferred",
    "good to have": "preferred", "nice to have": "preferred",
    "bonus": "bonus", "bonus points": "bonus", "bonus points if you have": "bonus",
    "about": "ignore", "about us": "ignore", "about the company": "ignore",
    "company": "ignore", "who we are": "ignore", "benefits": "ignore",
    "perks": "ignore", "compensation": "ignore", "equal opportunity": "ignore",
}

RESUME_HEADINGS = {
    "experience": "experience", "professional experience": "experience",
    "projects": "projects", "selected projects": "projects",
    "technical skills": "skills", "skills": "skills",
    "education": "education", "summary": "summary",
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "have", "in",
    "into", "is", "it", "of", "on", "or", "our", "that", "the", "their", "them", "these",
    "this", "to", "using", "we", "will", "with", "you", "your",
}


class ScoringError(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OllamaEmbedder:
    def __init__(self, base_url, model, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts):
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/embed", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                embeddings = json.load(response).get("embeddings", [])
        except (urllib.error.URLError, TimeoutError) as error:
            raise ScoringError(f"Ollama embedding request failed: {error}") from error
        if len(embeddings) != len(texts):
            raise ScoringError("Ollama returned an unexpected number of embeddings")
        return embeddings


def extract_pdf_text(path, timeout=30):
    try:
        result = subprocess.run(
            ["pdftotext", str(path), "-"], capture_output=True, check=True,
            text=True, timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ScoringError(f"Could not extract resume text: {error}") from error
    text = "\n".join(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not text:
        raise ScoringError("Resume PDF produced no text")
    return text


def text_chunks(text, limit=64):
    pieces = re.split(r"(?:\n+|(?<=[.!?])\s+|\s*[•◆❖]\s*)", text)
    chunks = []
    for piece in pieces:
        normalized = " ".join(piece.split())
        while len(normalized) > CHUNK_TEXT_LIMIT:
            boundary = normalized.rfind(" ", 0, CHUNK_TEXT_LIMIT + 1)
            boundary = boundary if boundary > 0 else CHUNK_TEXT_LIMIT
            chunks.append(normalized[:boundary])
            normalized = normalized[boundary:].lstrip()
        if len(normalized) >= 20:
            chunks.append(normalized)
    return chunks[:limit] or [" ".join(text.split())]


def _normalized_heading(text):
    value = text.strip().rstrip(":").replace("’", "'").lower()
    return " ".join(value.split())


def _heading_group(line):
    normalized = _normalized_heading(line)
    if normalized in SECTION_HEADINGS:
        return SECTION_HEADINGS[normalized]
    if len(normalized) > 80:
        return None
    if normalized.startswith("about ") and normalized not in {"about the role", "about the job"}:
        return "ignore"
    if len(normalized) <= 60:
        for heading in ("bonus points", "preferred qualifications", "minimum qualifications"):
            if normalized.startswith(f"{heading} "):
                return SECTION_HEADINGS[heading]
    return None


def _criterion_group(text, default):
    lowered = text.lower()
    if re.search(r"\b(required|must|minimum)\b", lowered):
        return "required"
    if re.search(r"\b(preferred|good to have|nice to have|bonus)\b", lowered):
        return "bonus" if default == "bonus" else "preferred"
    return default


def parse_job_criteria(text):
    criteria = []
    current_group = None
    recognized_heading = False
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            continue
        heading_group = _heading_group(line)
        if heading_group is not None:
            current_group = heading_group
            recognized_heading = True
            continue
        if current_group in {None, "ignore"}:
            continue
        line = re.sub(r"^[\s\-–—•◆❖*]+", "", line).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if len(sentence) >= 20:
                criteria.append({"text": sentence, "group": _criterion_group(sentence, current_group)})

    if not criteria and not recognized_heading:
        criteria = [{"text": chunk, "group": "core"} for chunk in text_chunks(text, MAX_CRITERIA)]
    return criteria[:MAX_CRITERIA]


def parse_resume_evidence(text):
    evidence = []
    section = "profile"
    buffered = ""
    buffered_bullet = False

    def flush():
        nonlocal buffered, buffered_bullet
        normalized = " ".join(buffered.split())
        if normalized:
            evidence.append({"text": normalized, "section": section})
        buffered = ""
        buffered_bullet = False

    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = " ".join(raw_line.split())
        if not line:
            continue
        heading = RESUME_HEADINGS.get(_normalized_heading(line))
        if heading:
            flush()
            section = heading
            continue
        bullet = re.match(r"^[\-–—•◆❖*]+\s*(.*)$", line)
        if bullet:
            flush()
            buffered = bullet.group(1).strip()
            buffered_bullet = True
        elif buffered_bullet and not re.search(r"[.!?]$", buffered):
            buffered = f"{buffered} {line}"
        else:
            flush()
            buffered = line
    flush()
    return evidence[:MAX_EVIDENCE] or [{"text": " ".join(text.split()), "section": "profile"}]


def cosine(left, right):
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def normalized_similarity(value):
    return round(max(0.0, min(100.0, (value - 0.30) / 0.45 * 100)), 1)


def score_band(score):
    return "Strong" if score >= 75 else "Moderate" if score >= 55 else "Weak"


def explicit_skills(text):
    lowered = text.lower()
    found = set()
    for canonical, aliases in SKILLS.items():
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lowered) for alias in aliases):
            found.add(canonical)
    return found


def _tokens(text):
    values = re.findall(r"c\+\+|[a-z0-9]+(?:[+#.-][a-z0-9]+)*", text.lower())
    return {value for value in values if len(value) > 2 and value not in STOP_WORDS}


def lexical_overlap(criterion, evidence):
    expected = _tokens(criterion)
    if not expected:
        return 0.0
    return round(len(expected & _tokens(evidence)) / len(expected) * 100, 1)


def _degree_match(criterion, resume_text):
    lowered = criterion.lower()
    resume = resume_text.lower()
    if re.search(r"\b(bachelor|bachelor's|bachelors)\b", lowered):
        return 100.0 if re.search(r"\b(bachelor|b\.?[es]\.?|btech|b\.tech)\b", resume) else 0.0
    if re.search(r"\b(master|master's|masters)\b", lowered):
        return 100.0 if re.search(r"\b(master|m\.?s\.?|mtech|m\.tech)\b", resume) else 0.0
    if re.search(r"\b(phd|ph\.d|doctorate)\b", lowered):
        return 100.0 if re.search(r"\b(phd|ph\.d|doctorate)\b", resume) else 0.0
    return None


class ResumeScorer:
    def __init__(self, store, embedder, scorer_version, pdf_extractor=extract_pdf_text, file_hasher=file_hash):
        self.store = store
        self.embedder = embedder
        self.scorer_version = scorer_version
        self.pdf_extractor = pdf_extractor
        self.file_hasher = file_hasher

    def _embeddings(self, content_hash, texts):
        texts_hash = hashlib.sha256(json.dumps(texts, ensure_ascii=True).encode()).hexdigest()
        cache_key = hashlib.sha256(f"{content_hash}|{self.embedder.model}|{texts_hash}".encode()).hexdigest()
        cached = self.store.get_cached_embeddings(cache_key)
        if cached is not None:
            return cached
        embeddings = self.embedder.embed(texts)
        self.store.save_cached_embeddings(cache_key, content_hash, self.embedder.model, texts_hash, embeddings)
        return embeddings

    def score(self, scoring_input):
        job_text = scoring_input["job_description"]
        if self.file_hasher(scoring_input["resume_path"]) != scoring_input["resume_content_hash"]:
            raise ScoringError("Resume PDF no longer matches the captured content hash")
        resume_text = self.pdf_extractor(scoring_input["resume_path"])
        criteria = parse_job_criteria(job_text)
        evidence = parse_resume_evidence(resume_text)
        if not criteria:
            raise ScoringError("Job description produced no scoreable criteria")

        criterion_vectors = self._embeddings(
            scoring_input["job_description_hash"], [item["text"] for item in criteria]
        )
        evidence_vectors = self._embeddings(
            scoring_input["resume_content_hash"], [item["text"] for item in evidence]
        )
        resume_skills = explicit_skills(resume_text)
        scored_criteria = []
        for criterion, criterion_vector in zip(criteria, criterion_vectors):
            pair_scores = []
            for item, evidence_vector in zip(evidence, evidence_vectors):
                semantic = normalized_similarity(cosine(criterion_vector, evidence_vector))
                lexical = lexical_overlap(criterion["text"], item["text"])
                combined = max(semantic, round(semantic * 0.65 + lexical * 0.35, 1))
                pair_scores.append((combined, item["text"]))
            match_score, best_evidence = max(pair_scores, key=lambda item: item[0])
            criterion_skills = explicit_skills(criterion["text"])
            if criterion_skills:
                skill_coverage = len(criterion_skills & resume_skills) / len(criterion_skills) * 100
                match_score = round(match_score * 0.75 + skill_coverage * 0.25, 1)
            degree_score = _degree_match(criterion["text"], resume_text)
            if degree_score is not None:
                match_score = round(match_score * 0.70 + degree_score * 0.30, 1)
            scored_criteria.append({**criterion, "score": match_score, "evidence": best_evidence})

        categories = {}
        for group in (*GROUP_WEIGHTS, "bonus"):
            values = [item["score"] for item in scored_criteria if item["group"] == group]
            if values:
                categories[group] = round(sum(values) / len(values), 1)
        active_weights = {group: weight for group, weight in GROUP_WEIGHTS.items() if group in categories}
        total_weight = sum(active_weights.values())
        if not total_weight:
            raise ScoringError("Job description produced no primary criteria")
        base_score = sum(categories[group] * weight for group, weight in active_weights.items()) / total_weight
        score = round(min(100.0, base_score + categories.get("bonus", 0.0) / 100 * 3.0), 1)
        job_skills = explicit_skills(job_text)
        gaps = sorted(
            (item for item in scored_criteria if item["group"] != "bonus" and item["score"] < 60),
            key=lambda item: (item["score"], item["group"]),
        )[:3]
        return {
            "application_key": scoring_input["application_key"],
            "input_fingerprint": score_input_fingerprint(
                scoring_input, self.scorer_version, self.embedder.model
            ),
            "scorer_version": self.scorer_version,
            "embedding_model": self.embedder.model,
            "score": score,
            "band": score_band(score),
            "matched_terms": sorted(job_skills & resume_skills, key=str.lower),
            "category_breakdown": categories,
            "top_gaps": [
                {"criterion": item["text"], "group": item["group"], "score": item["score"]}
                for item in gaps
            ],
        }


def score_input_fingerprint(
    scoring_input, scorer_version, embedding_model, deployment_revision=""
):
    source = "|".join((
        scoring_input["application_key"], scoring_input["resume_content_hash"],
        scoring_input["job_description_hash"], scorer_version, embedding_model,
        deployment_revision,
    ))
    return hashlib.sha256(source.encode()).hexdigest()


def score_cache_key(scoring_input, scorer_version, embedding_model, deployment_revision=""):
    source = "|".join((
        scoring_input["role_company_key"], scoring_input["resume_content_hash"],
        scorer_version, embedding_model, deployment_revision,
    ))
    return hashlib.sha256(source.encode()).hexdigest()


def role_company_key(company, job_title):
    normalized = "|".join(
        " ".join(str(value).strip().casefold().split())
        for value in (company, job_title)
    )
    return hashlib.sha256(normalized.encode()).hexdigest()
