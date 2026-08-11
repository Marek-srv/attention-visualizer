# Manual deployment: GitHub, Render, and Vercel

This repository deploys as two services:

- `backend` is a FastAPI web service on Render.
- `frontend_1` is a Vite static frontend on Vercel.

Deploy the backend first because the frontend build needs its public URL. Then deploy the frontend, add its final origin to the backend's CORS allowlist, and verify a browser request end to end.

## 1. Upload a clean source tree to GitHub

Open the existing empty GitHub repository, `Marek-srv/attention-visualizer`. On its empty-repository page, select **uploading an existing file**; after the first batch, use **Add file → Upload files** for later batches. Do not add a GitHub-generated README, `.gitignore`, or license because this project already supplies its own root files.

Keep `render.yaml`, `backend`, `frontend_1`, and the other root files directly at the GitHub repository root. Do not add an extra enclosing `attention-visualizer/` directory or the deployment root-directory settings will no longer match.

The GitHub browser uploader has two relevant limits: [25 MiB per file and 100 files per upload](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository#adding-a-file-to-a-repository-on-github). This clean source tree contains more than 100 files, so upload it in separate commits/batches. For example, upload the root files plus `backend`, then `frontend_1`, then `frontend`. The bundled checkpoint is only about 425 KB and is below the per-file limit.

The browser uploader uploads what you select; it does not use the local Git index to filter a folder. If you drag folders, first make a separate clean staging copy and remove generated/private content from that copy. Do not upload the staging folder as a `.zip` file—GitHub will store the archive instead of expanding the project tree.

Exclude all of the following:

- `.git/`, `.venv/`, `backend/.venv/`, and any other virtual environment
- every `node_modules/` and `dist/` directory
- `.env` and any file containing credentials, tokens, or private URLs
- `__pycache__/`, `.pytest_cache/`, `.pytest_tmp/`, and other generated caches

Make sure these easy-to-miss files are included:

- `.gitignore`, `render.yaml`, `README.md`, and `DEPLOYMENT.md`
- `backend/.python-version` and `backend/requirements.txt`
- `backend/checkpoints/tiny_transformer_best.pt`
- `backend/checkpoints/tiny_transformer_best.json`
- `frontend_1/.env.example`, `frontend_1/package.json`, `frontend_1/package-lock.json`, and `frontend_1/vercel.json`

After all batches are committed, verify that GitHub shows `backend/app/main.py` and `frontend_1/package.json` at exactly those paths.

## 2. Deploy the backend to Render

1. Sign in to Render and choose **New → Blueprint**.
2. Connect the GitHub repository and allow Render to read it.
3. Select the repository. Render should detect the root `render.yaml` and show one free Python web service named `attention-visualizer-api`.
4. Apply the Blueprint and wait for the deploy to become **Live**. The Blueprint uses `backend` as its root directory, Python 3.11.11, a CPU-only PyTorch wheel followed by `pip install -r requirements.txt`, and the required `$PORT` binding. The CPU wheel avoids downloading unused CUDA packages on the small free instance.
5. Copy the service origin shown by Render, for example `https://attention-visualizer-api.onrender.com`. Your actual name may have a suffix. Keep only the origin: no trailing slash and no `/api`.
6. Open the health URL:

   ```text
   https://YOUR-RENDER-SERVICE.onrender.com/api/health
   ```

   A healthy backend returns:

   ```json
   {"status":"ok","service":"attention-visualizer-api"}
   ```

FastAPI's interactive API documentation is at `https://YOUR-RENDER-SERVICE.onrender.com/docs`.

## 3. Deploy `frontend_1` to Vercel

1. In Vercel, choose **Add New → Project** and import the same GitHub repository.
2. Set **Root Directory** to `frontend_1`. Do not select the repository root or the older `frontend` directory.
3. Vercel should detect Vite. The checked-in `frontend_1/vercel.json` sets `npm run build` and the `dist` output directory.
4. Before deploying, add this Production environment variable:

   ```text
   VITE_API_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com
   ```

   Use the Render origin only—no trailing slash and no `/api`. `VITE_` variables are compiled into the frontend at build time.
5. Deploy and copy the stable production origin, for example `https://YOUR-PROJECT.vercel.app`. Keep it without a trailing slash or path.

At this point the page can load, but cross-origin API calls will remain blocked until the backend allows the exact Vercel origin.

## 4. Allow the Vercel origin on Render

1. Open the Render backend service and go to **Environment**.
2. Add:

   ```text
   CORS_ORIGINS=https://YOUR-PROJECT.vercel.app
   ```

3. For more than one fixed frontend origin, use a comma-separated list with no paths, for example:

   ```text
   CORS_ORIGINS=https://YOUR-PROJECT.vercel.app,https://preview.example.com
   ```

4. Save the variable and wait for the resulting Render redeploy to become **Live**. If Render does not start one automatically, choose **Manual Deploy → Deploy latest commit**.
5. If `VITE_API_BASE_URL` was added or corrected after the Vercel deployment, redeploy Vercel too; changing a build-time variable does not update an already-built deployment.

Vercel preview deployments use different origins. Add a specific preview origin to `CORS_ORIGINS` only when you need that preview; do not use a wildcard with credentialed CORS.

## 5. Check the deployed application

1. Reopen `https://YOUR-RENDER-SERVICE.onrender.com/api/health` and confirm the JSON response.
2. Open the Vercel production URL and exercise **Toy Math Lab** or another feature that calls the API.
3. To test CORS directly, open the browser developer console while on the Vercel site and run:

   ```js
   fetch("https://YOUR-RENDER-SERVICE.onrender.com/api/health")
     .then((response) => response.json())
     .then(console.log);
   ```

   It should log the health object without a CORS error. Opening the Render URL directly proves that the backend is alive, but this browser-console check proves that the Vercel origin is allowed.
4. If a request fails, inspect the Vercel browser Network panel and the Render service logs. Confirm the two environment values contain origins only and that `CORS_ORIGINS` exactly matches the Vercel page's origin.

## Render free-service limitations

Render's [free web-service documentation](https://render.com/docs/free) currently states that a free service spins down after 15 minutes without inbound traffic. Its next request can take about a minute while the service wakes up.

The free service filesystem is ephemeral. A checkpoint created or overwritten by web training can disappear whenever the service spins down, restarts, or redeploys, and free web services cannot attach a persistent disk. The small checkpoint committed at `backend/checkpoints/tiny_transformer_best.pt` and its JSON metadata are restored from the GitHub source on a fresh deploy, but any new training performed on Render is temporary. Treat hosted training as a short demonstration, not durable model storage; use external object storage or a paid persistent design before relying on trained output.

Free instances can also restart unexpectedly and have limited CPU and memory. Long training runs and optional pretrained-model downloads may be slow, interrupted, or exceed the available resources.
