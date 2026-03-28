# DeepGuard Deployment Guide

This guide will walk you through securing your project, pushing it to GitHub, and deploying it live on Vercel.

---

## 🔒 Step 1: Secure Your Project (API Keys & Secrets)

Your frontend connects to a backend, and eventually, you might add API keys or secrets directly into your frontend code using a `.env` file. We need to make sure these **never** get uploaded to GitHub.

1. **Check ignored files:** We have already updated your `.gitignore` file to include `.env`, `.env.*`, and your `backend-repo/` folder. This means any file named `.env` will be completely ignored by Git.
2. **Double check:** Make sure you **do not** have any hardcoded API keys left in files like `App.jsx`, `UploadCard.jsx`, etc. (I have checked the codebase, and currently, there are no hardcoded API keys in the frontend source).

> [!IMPORTANT]
> Because `.env` is ignored by Git, when you deploy to Vercel, you will need to manually paste your environment variables into the Vercel dashboard.

---

## 🐙 Step 2: Push to GitHub

Now that the project is secure, let's push the frontend code to a new GitHub repository.

1. **Create a new repository on GitHub** (e.g., name it `deepguard-frontend`). Do not initialize it with a README or .gitignore (leave it completely empty).
2. **Open your terminal** and navigate to your frontend folder:
   ```bash
   cd c:\Users\Shree\OneDrive\Desktop\DEEPGUARD\deepguard-frontend
   ```
3. **Initialize Git & Push:**
   Run the following commands one by one (replace `<your-username>` with your actual GitHub username):
   ```bash
   git init
   git add .
   git commit -m "Initial commit for DeepGuard frontend"
   git branch -M main
   git remote add origin https://github.com/<your-username>/deepguard-frontend.git
   git push -u origin main
   ```

---

## 🚀 Step 3: Deploy to Vercel

Vercel is the easiest and fastest way to deploy a Vite + React application.

1. **Go to [Vercel](https://vercel.com/)** and log in with your GitHub account.
2. Click the **"Add New Project"** button.
3. Find your newly created `deepguard-frontend` repository in the list and click **Import**.
4. **Configure your project:**
   - **Framework Preset:** Vercel should automatically detect **Vite**. If not, select it from the dropdown.
   - **Root Directory:** Leave it as `./` (the default).
   - **Environment Variables:** If you later add any API keys (like a Supabase key, Firebase config, etc.) that your frontend uses via `import.meta.env`, open the **Environment Variables** tab here and paste them in.
5. Click **Deploy**.

Vercel will now install your packages, build the Vite app, and assign it a live URL!

---

## 🛠️ Step 4: Update Backend CORS (If Needed)

Currently, your backend (`https://pre-hackathon-jklu.onrender.com`) allows all origins (`allow_origins=["*"]`). 
This is great for the hackathon because your Vercel URL will instantly be able to communicate with your backend without any CORS errors.

However, for a production environment later, you should edit your backend `main.py` to only allow your specific Vercel URL:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app-name.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🎉 You're Live!
Once Vercel finishes, you will get a link like `https://deepguard.vercel.app`. Your site is now live on the internet!
