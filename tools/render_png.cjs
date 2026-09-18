// Render an SVG file to PNG with headless Chromium (Playwright).
// Usage: node tools/render_png.cjs figure.svg figure.png [scale]
// Needs the `playwright` package resolvable (e.g. NODE_PATH=/path/to/node_modules)
// and a Chromium install (PLAYWRIGHT_BROWSERS_PATH). The SVG is the source of
// truth; the PNG is a convenience for viewers that do not render SVG.
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const [svgPath, pngPath, scaleArg] = process.argv.slice(2);
  if (!svgPath || !pngPath) {
    console.error("usage: node tools/render_png.cjs in.svg out.png [scale]");
    process.exit(2);
  }
  const scale = Number(scaleArg || 2);
  const svg = fs.readFileSync(svgPath, "utf8");
  const browser = await chromium.launch();
  const page = await browser.newPage({ deviceScaleFactor: scale });
  await page.setContent(`<!doctype html><html><body style="margin:0;background:#f2f2f2">${svg}</body></html>`);
  const el = await page.$("svg");
  await el.screenshot({ path: pngPath, omitBackground: false });
  await browser.close();
  console.log(`wrote ${path.relative(process.cwd(), pngPath)} (${scale}x)`);
})();
