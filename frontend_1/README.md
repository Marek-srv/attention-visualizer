# Attention Lab Interactive

`frontend_1` is an experimental, spatial Transformer explainer that runs beside the original application. It uses the existing FastAPI endpoints and does not replace either `frontend` or `backend`.

## Run on Windows PowerShell

Start the backend:

```powershell
cd C:\Users\sriram.vishal\Documents\attention-visualizer\backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --port 8000
```

If your working environment is the repository-level `.venv`, use this instead from `backend`:

```powershell
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

In another terminal:

```powershell
cd C:\Users\sriram.vishal\Documents\attention-visualizer\frontend_1
npm install
npm run dev
```

Open <http://localhost:5174>. The original frontend remains available at <http://localhost:5173> when its own development server is running.

## Learning modes

- **Toy Math Lab** uses the fixed four-dimensional, post-normalization values from `POST /api/inspect`. Its advanced drawer edits Q/K/V, WO, normalization, and feed-forward parameters, then recalculates on the backend.
- **Train Model** starts and monitors the local tiny decoder training job. It polls only while a job is running.
- **Predict** loads the saved local checkpoint, then provides next-token probabilities and short greedy or seeded generation.
- **Trained Model Inspector** displays selected tensors from the locally trained pre-normalization model.
- **Real Model** checks status without loading anything. The configured pretrained model is downloaded or read from cache only after **Load model** is selected.

Toy values, locally trained values, and pretrained values are labelled separately. Model probabilities are not factual confidence, and an attention pattern does not completely explain a prediction.

## Guided and Explore modes

**Guided** moves through the forward pass in order. Use Previous, Play/Pause, Next, Restart, and the speed selector. Playback stops at Next Token and does not loop.

**Explore** lets you choose any available stage and token directly. Opening a stage keeps the rest of the model in context. Close it with **Back**, by selecting the active stage again, or with `Escape`.

## Keyboard use

- Use `Tab` and `Shift+Tab` to move through modes, tokens, stages, matrices, and controls.
- Press `Enter` or `Space` to select a focused token, stage, heatmap cell, or action.
- Press `Escape` to close an expanded stage or textbook drawer.
- Every SVG visualization has a parallel labelled control, table, or inspector so exact values do not require pointer input.

## Production and checks

```powershell
npm run typecheck
npm run test -- --run
npm run lint
npm run build
```

Set `VITE_API_BASE_URL` for a production API hosted on another origin. When it is absent, requests use relative `/api` URLs. Development proxies `/api` to `http://127.0.0.1:8000`.

### Deploy to Vercel

Import the GitHub repository in the Vercel dashboard, then configure the project as follows:

- Set **Root Directory** to `frontend_1` and keep the **Framework Preset** as Vite.
- Set **Build Command** to `npm run build` and **Output Directory** to `dist`.
- Add `VITE_API_BASE_URL` under **Settings > Environment Variables**, using the Render backend origin (for example, `https://your-render-service.onrender.com`) without `/api`.
- Redeploy after adding or changing the environment variable because Vite embeds it during the build.

For local development, leave `VITE_API_BASE_URL` unset so the existing Vite proxy continues to send `/api` requests to `http://127.0.0.1:8000`.

## Attribution

The interaction design is independently implemented in React and informed by [Transformer Explainer](https://github.com/poloclub/transformer-explainer), Cho et al. No source code, model assets, screenshots, branding, or educational prose from that project are bundled here. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
