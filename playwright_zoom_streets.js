const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1600, height: 900 });

  await page.goto('file:///d:/Antigravity/Tesis/Tesis_Cariari/resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html');

  // Esperar que el canvas Three.js esté presente
  await page.waitForSelector('canvas', { timeout: 30000 });
  await page.waitForTimeout(5000);  // dar tiempo al script para ejecutar

  // Vista 1: ángulo bajo desde el sur, para ver profundidad de canales
  await page.evaluate(() => {
    if (typeof camera !== 'undefined' && typeof controls !== 'undefined') {
      camera.position.set(-400, 950, 900);
      controls.target.set(0, 920, 0);
      controls.update();
    }
  });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'output/playwright/calles-dep-inicial.png' });
  console.log('Screenshot inicial guardado');

  // Vista 2: nadir (desde arriba) para ver la traza de calles
  await page.evaluate(() => {
    if (typeof camera !== 'undefined' && typeof controls !== 'undefined') {
      camera.position.set(0, 2200, 10);
      controls.target.set(0, 920, 0);
      controls.update();
    }
  });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'output/playwright/calles-dep-nadir.png' });
  console.log('Screenshot nadir guardado');

  // Vista 3: zoom a zona urbana central, ángulo lateral bajo
  await page.evaluate(() => {
    if (typeof camera !== 'undefined' && typeof controls !== 'undefined') {
      camera.position.set(100, 955, 350);
      controls.target.set(80, 920, -80);
      controls.update();
    }
  });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: 'output/playwright/calles-dep-zoom.png' });
  console.log('Screenshot zoom guardado');

  await browser.close();
})();
