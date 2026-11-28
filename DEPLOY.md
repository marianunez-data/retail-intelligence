# Deployment

The same Docker image runs on every platform. It installs the package, trains the retention model from the committed dbt snapshots in `data/processed/`, copies the committed demand forecast and pricing results, and serves the Streamlit app on port 8501. Training inside the image guarantees that the model is trained and served with identical library versions. No raw data is needed on the server.

Test it locally first (needs Docker):

    make docker        # builds the image and runs it on http://localhost:8501

---

## A. Streamlit Community Cloud (free, sign in with GitHub)

1. Go to https://share.streamlit.io and **Continue with GitHub**.
2. **Create app**, then **Deploy a public app from GitHub**:
   - Repository: `marianunez-data/retail-intelligence`, branch `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: choose a name, for example `retail-intelligence`
3. **Advanced settings**: Python **3.12**, and paste the secrets:

        LLM_API_KEY = "your-gemini-key"
        LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
        LLM_MODEL = "gemini-3.5-flash-lite"

4. **Deploy**. Streamlit installs `requirements.txt` (this package with the app dependencies) in a few minutes. On the first visit the app trains the retention model in about three seconds. Root-level secrets are exposed as environment variables, which is what the AI analyst reads.

Free apps sleep after several days without visitors; the first visitor wakes them with one click. Every push to `main` redeploys automatically.

---

## B. Hugging Face Spaces (free, no card)

1. Create an account on https://huggingface.co, then **New Space**: name `retail-intelligence`, SDK **Docker**, blank template, hardware **CPU basic (free)**, visibility **Public**.
2. In the Space, **Settings, Variables and secrets**:
   - Secret `LLM_API_KEY` = your Gemini key (section D)
   - Variable `LLM_BASE_URL` = `https://generativelanguage.googleapis.com/v1beta/openai/`
   - Variable `LLM_MODEL` = `gemini-3.8-flash`
3. Create an access token with **Write** permission (**Settings, Access Tokens**).
4. From the repository on your computer:

        pip install -U huggingface_hub
        hf auth login                                  # paste the write token
        bash deploy/prepare_hf_space.sh ../hf-space
        hf upload YOUR_HF_USER/retail-intelligence ../hf-space . --repo-type space --commit-message "Deploy Retail Intelligence"

   The script copies only what the image needs and uses `deploy/HF_SPACE_README.md` as the Space README, which tells Hugging Face to serve port 8501. `hf upload` handles binary files such as the Parquet forecast history, which a plain `git push` to the Hub rejects.
5. The build takes about 5 to 8 minutes (**Logs** tab). The app is then live at `https://huggingface.co/spaces/YOUR_HF_USER/retail-intelligence`. Free Spaces sleep after a period without visitors and wake up on the next visit in about a minute.

To update after a code change, run steps 4 again.

---

## C. Oracle Cloud (OCI), always on

Needs a card at sign-up for identity checks. Use **Always Free** resources only: they keep running after the 30-day trial credit ends. Since 18 August 2026 the Always Free Ampere A1 allowance is 2 OCPUs and 12 GB memory in total; this app needs 1 OCPU and 6 GB, which leaves room for a second small VM.

### 1. Create the VM (10 minutes)
1. OCI Console, menu, **Compute, Instances, Create instance**.
2. Name: `retail-intelligence`.
3. Image: **Canonical Ubuntu 22.04** (or 24.04).
4. Shape: **Ampere, VM.Standard.A1.Flex**, 1 OCPU, 6 GB memory (look for the "Always Free-eligible" label).
5. Networking: create a new VCN with a **public subnet** and keep **Assign a public IPv4 address** on.
6. SSH keys: **Save private key** and keep the file safe.
7. Create, then copy the **Public IP** once the instance is Running.

If you see "Out of capacity", try another availability domain, retry later, or use **VM.Standard.E2.1.Micro** (also Always Free, 1 GB memory, slower build).

### 2. Open port 8501 (3 minutes)
1. On the instance page, open the **Subnet**, then the **Default Security List**.
2. **Add Ingress Rules**: Source CIDR `0.0.0.0/0`, IP protocol TCP, destination port `8501`. Save.

The setup script opens the same port in the VM's own firewall.

### 3. Connect and deploy
From your terminal:

    chmod 400 ~/Downloads/ssh-key-*.key
    ssh -i ~/Downloads/ssh-key-*.key ubuntu@YOUR_PUBLIC_IP

On the VM:

    git clone https://github.com/marianunez-data/retail-intelligence.git
    cd retail-intelligence
    cp .env.example .env
    nano .env                  # paste LLM_API_KEY, save with Ctrl+O, exit with Ctrl+X
    bash deploy/oci_setup.sh

The script installs Docker, opens the port, builds the image (about 5 minutes on 1 OCPU) and starts the container with automatic restart. It prints the URL, `http://YOUR_PUBLIC_IP:8501`. Put it in the README.

`.env` stays on the VM; it is listed in `.gitignore`, so the key never reaches GitHub.

### Update after a code change

    cd ~/retail-intelligence && git pull && bash deploy/oci_setup.sh

---

## D. Free LLM key for the AI analyst

1. Go to https://aistudio.google.com and sign in with a Google account.
2. **Get API key**, create a key. The free tier needs no card.
3. Use it as `LLM_API_KEY`. The default model is `gemini-3.8-flash`; if your account does not offer it or its free quota runs out, set `LLM_MODEL` to `gemini-3.5-flash-lite` or any current Flash model ID listed in AI Studio.

Any OpenAI-compatible provider works by changing the three variables, with no code changes. Without a key the dashboard works fully and the AI analyst page shows a "not configured" notice. The free tier has per-minute and per-day request limits, which are enough for a portfolio demo.

---

## E. AWS (optional reference)

`reference/aws/` contains a SageMaker notebook, a Lambda function and a static web form for serving the model through API Gateway. It needs an AWS account and is not part of CI.
