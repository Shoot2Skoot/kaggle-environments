import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('renders the board', async ({ page }) => {
  await expect(page.locator('.renderer-container')).toBeVisible();
  await expect(page.locator('.sh-title')).toHaveText(/Secret Hitler/);
  await expect(page.locator('.sh-track').first()).toBeVisible();
  await expect(page.locator('.sh-players .sh-player').first()).toBeVisible();
});

test('displays state at mid-game', async ({ page }) => {
  const slider = page.locator('input[type="range"]');
  await slider.waitFor({ state: 'visible' });
  const maxValue = await slider.getAttribute('max');
  const midStep = Math.floor(parseInt(maxValue || '0') / 2);
  await slider.fill(String(midStep));
  await page.waitForTimeout(200);
  await expect(page.locator('.sh-board')).toBeVisible();
  await expect(page.locator('.sh-slot')).toHaveCount(11); // 5 liberal + 6 fascist slots
});

test('shows a winner at the final step', async ({ page }) => {
  const slider = page.locator('input[type="range"]');
  await slider.waitFor({ state: 'visible' });
  const maxValue = await slider.getAttribute('max');
  await slider.fill(maxValue || '0');
  await page.waitForTimeout(200);
  await expect(page.locator('.sh-winner')).toHaveText(/Liberal|Fascist/);
});
