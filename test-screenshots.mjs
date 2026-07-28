import puppeteer from 'puppeteer';
import { exec, spawn } from 'child_process';
import { promisify } from 'util';
import fs from 'fs';
import path from 'path';
import http from 'http';

const SCREENSHOT_DIR = path.join(process.cwd(), 'test-screenshots');
const DEMOS = [
  'GLB Viewer',
  'PBR + Shadow',
  'Toon (Multi-Material)',
  'Arc Toon',
  'Grass',
  'Water',
  'ShellFur',
  'Particles',
  'Boid',
  'MeshGen',
  'IndirectDraw',
  'GlitchedMosaics',
  'GreedySnake',
  'FractalNoise',
  'Meteorograph',
  'Toon Textured',
  'Terrain Hydrology',
  'ACES Pipeline',
  'Hair/Eye Shadow',
  'Showcase',
];

const sleep = (ms: number) => new Promise(r => setTimeout(r, ms));

async function waitForServer(url: string, timeoutMs = 30000): Promise<boolean> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      await new Promise<void>((resolve, reject) => {
        http.get(url, res => { res.resume(); resolve(); }).on('error', reject);
      });
      return true;
    } catch { await sleep(500); }
  }
  return false;
}

function getImageStats(filePath: string): {
  size: number;
  isBlack: boolean;
  hasContent: boolean;
} {
  const size = fs.statSync(filePath).size;
  return {
    size,
    isBlack: size < 3000,
    hasContent: size > 5000,
  };
}

async function run() {
  if (!fs.existsSync(SCREENSHOT_DIR)) fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

  console.log('Starting dev server...');
  const server = spawn('npx', ['vite', '--port', '5199'], {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: true,
  });
  server.stdout?.on('data', (d: Buffer) => process.stdout.write(d));
  server.stderr?.on('data', (d: Buffer) => process.stderr.write(d));

  const ready = await waitForServer('http://localhost:5199');
  if (!ready) { console.error('Server failed to start'); server.kill(); process.exit(1); }
  console.log('Dev server ready on port 5199');

  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,UseSkiaRenderer',
      '--use-vulkan',
      '--enable-webgpu-developer-features',
    ],
    defaultViewport: { width: 1280, height: 720 },
  });

  const page = await browser.newPage();

  page.on('console', msg => {
    if (msg.type() === 'error') console.log(`  [console.error] ${msg.text()}`);
  });
  page.on('pageerror', err => console.log(`  [page error] ${err.message}`));

  const results: { demo: string; size: number; passed: boolean; issues: string[] }[] = [];

  await page.goto('http://localhost:5199', { waitUntil: 'networkidle0' });
  await sleep(2000);

  for (const demo of DEMOS) {
    const issues: string[] = [];
    const safeName = demo.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();

    try {
      await page.evaluate((name: string) => {
        const sel = document.querySelector('select') as HTMLSelectElement;
        if (!sel) return;
        for (let i = 0; i < sel.options.length; i++) {
          if (sel.options[i].text.includes(name)) {
            sel.selectedIndex = i;
            sel.dispatchEvent(new Event('change'));
            return;
          }
        }
      }, demo);

      await sleep(3000);

      const shotPath = path.join(SCREENSHOT_DIR, `${safeName}.png`);
      await page.screenshot({ path: shotPath, type: 'png' });

      const stats = getImageStats(shotPath);
      if (stats.isBlack) issues.push(`Black image (${stats.size} bytes)`);
      if (!stats.hasContent) issues.push(`Low content (${stats.size} bytes)`);

      console.log(`  ${issues.length === 0 ? '✅' : '❌'} ${demo}: ${stats.size} bytes`);
    } catch (e: any) {
      issues.push(`Error: ${e.message}`);
      console.log(`  ❌ ${demo}: ${e.message}`);
    }

    results.push({ demo, size: results.length, passed: issues.length === 0, issues });
  }

  await browser.close();
  server.kill();

  console.log('\n' + '='.repeat(60));
  console.log('SCREENSHOT ACCEPTANCE RESULTS');
  console.log('='.repeat(60));

  let pass = 0;
  for (const r of results) {
    const icon = r.passed ? '✅' : '❌';
    console.log(`${icon} ${r.demo}`);
    if (!r.passed) r.issues.forEach(i => console.log(`    ${i}`));
    else pass++;
  }
  console.log(`\n${pass}/${results.length} demos passed visual check`);
  console.log(`Screenshots saved to ${SCREENSHOT_DIR}/`);

  process.exit(pass === results.length ? 0 : 1);
}

run().catch(e => { console.error(e); process.exit(1); });
