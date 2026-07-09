# Reactive Particle Demo

Browser-only reactive particle demo. The app uses the webcam for hand and face tracking, then drives a Three.js particle field with MediaPipe landmarks.

## Stack

- Vite
- React UI shell
- Three.js WebGL particle engine
- MediaPipe Tasks Vision
- Cloudflare Workers static assets via Wrangler

## Scripts

```bash
npm install
npm run dev
npm run build
npm run verify:hand
npm run verify:face
```

## Deployment

The app builds to `dist/` and deploys with Wrangler using `wrangler.jsonc`.

```bash
npm run build
npx wrangler deploy
```

## Notes

- Webcam tracking runs fully in the browser.
- The React layer owns the HUD and settings UI.
- The Three.js and MediaPipe loops stay outside React to keep animation and tracking work off the React render path.
