# Testing Guide

Complete testing documentation for the Autoboros AI Advisory Board.

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Stack](#test-stack)
- [Running Tests](#running-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [E2E Tests](#e2e-tests)
- [Test Coverage](#test-coverage)
- [Mocking](#mocking)
- [CI Testing](#ci-testing)

---

## Testing Philosophy

- **Test behaviour, not implementation** — tests should verify what the system does, not how it does it
- **Fast feedback** — unit tests should run in milliseconds
- **Realistic scenarios** — integration tests should mirror real user flows
- **Confidence over coverage** — 80% meaningful coverage beats 100% shallow coverage

---

## Test Stack

| Tool | Purpose |
|------|---------|
| Vitest | Unit and integration test runner |
| React Testing Library | Component testing |
| Playwright | End-to-end testing |
| MSW | API mocking |
| @testing-library/user-event | Simulating user interactions |

---

## Running Tests

```bash
# All tests
npm run test

# Unit tests only
npm run test:unit

# Integration tests only
npm run test:integration

# E2E tests (requires running dev server)
npm run test:e2e

# Watch mode for TDD
npm run test:watch

# Coverage report
npm run test:coverage

# Type checking
npm run type-check
```

---

## Unit Tests

### Agent Analysis Tests

```javascript
// tests/unit/agents.test.js
import { describe, it, expect } from 'vitest';
import { generateEnhancedAnalysis, calculateOverallConfidence } from '../../src/utils/analysis';

describe('generateEnhancedAnalysis', () => {
  const ALL_AGENTS = [
    'primenode', 'devils-advocate', 'strategic-thinker',
    'financial-analyst', 'people-manager', 'innovation-catalyst', 'detail-tracker'
  ];

  it('returns valid structure for all agent types', () => {
    ALL_AGENTS.forEach(agentId => {
      const result = generateEnhancedAnalysis(agentId, 'Should we expand to a new market?');

      expect(result).toHaveProperty('summary');
      expect(result).toHaveProperty('details');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('priority');
      expect(result).toHaveProperty('timeframe');
      expect(result).toHaveProperty('actionItems');

      expect(typeof result.summary).toBe('string');
      expect(result.summary.length).toBeGreaterThan(0);
      expect(Array.isArray(result.details)).toBe(true);
      expect(result.details.length).toBeGreaterThan(0);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
      expect(['Low', 'Medium', 'High', 'Critical']).toContain(result.priority);
      expect(Array.isArray(result.actionItems)).toBe(true);
    });
  });

  it('returns different analysis for different agents', () => {
    const decision = 'Should we hire 10 more engineers?';
    const financial = generateEnhancedAnalysis('financial-analyst', decision);
    const people = generateEnhancedAnalysis('people-manager', decision);
    expect(financial.summary).not.toBe(people.summary);
  });

  it('handles empty decision string gracefully', () => {
    const result = generateEnhancedAnalysis('primenode', '');
    expect(result).toBeTruthy();
    expect(result.confidence).toBeGreaterThan(0);
  });
});

describe('calculateOverallConfidence', () => {
  it('returns average confidence across selected agents', () => {
    const activeAdvisors = ['primenode', 'financial-analyst'];
    const confidence = calculateOverallConfidence(activeAdvisors);
    // primenode: 0.94, financial-analyst: 0.96 → avg: 0.95
    expect(confidence).toBeCloseTo(0.95, 2);
  });

  it('returns 0 for empty advisor list', () => {
    expect(calculateOverallConfidence([])).toBe(0);
  });
});
```

### Decision Category Tests

```javascript
// tests/unit/categorize.test.js
import { describe, it, expect } from 'vitest';
import { determineCategory } from '../../src/utils/analysis';

describe('determineCategory', () => {
  it('categorizes financial decisions', () => {
    expect(determineCategory('Should we increase our budget?')).toBe('financial');
    expect(determineCategory('What is the cost of this initiative?')).toBe('financial');
  });

  it('categorizes strategic decisions', () => {
    expect(determineCategory('Should we enter the European market?')).toBe('strategic');
    expect(determineCategory('How do we compete with our rivals?')).toBe('strategic');
  });

  it('categorizes people decisions', () => {
    expect(determineCategory('Should we hire a new team lead?')).toBe('people');
    expect(determineCategory('How do we restructure the team?')).toBe('people');
  });

  it('falls back to operational for unclassified decisions', () => {
    expect(determineCategory('What should we do next?')).toBe('operational');
  });
});
```

### Export Utility Tests

```javascript
// tests/unit/export.test.js
import { describe, it, expect, vi } from 'vitest';
import { exportDecision, generateCSV, formatAsMarkdown } from '../../src/utils/export';

const mockDecision = {
  id: 123,
  text: 'Should we launch a new product?',
  timestamp: '2026-04-19T10:00:00Z',
  category: 'strategic',
  overallConfidence: 0.89,
  advisorInputs: [
    {
      advisor: 'Financial Analyst',
      advisorId: 'financial-analyst',
      confidence: 0.96,
      priority: 'High',
      analysis: 'Strong ROI projected.',
      detailedAnalysis: {
        summary: 'Strong ROI projected.',
        details: ['ROI of 127% over 24 months'],
        actionItems: ['Secure budget approval']
      }
    }
  ],
  treeOfThoughts: {
    rootNode: 'Should we launch a new product?',
    branches: []
  }
};

describe('formatAsMarkdown', () => {
  it('produces valid markdown output', () => {
    const md = formatAsMarkdown(mockDecision);
    expect(md).toContain('# Decision Analysis');
    expect(md).toContain('Should we launch a new product?');
    expect(md).toContain('Financial Analyst');
    expect(md).toContain('89%');
  });
});

describe('generateCSV', () => {
  it('includes all required columns', () => {
    const csv = generateCSV([mockDecision]);
    expect(csv).toContain('timestamp');
    expect(csv).toContain('category');
    expect(csv).toContain('confidence');
  });

  it('handles empty decision list', () => {
    const csv = generateCSV([]);
    expect(csv).toBeTruthy();
    expect(csv.split('\n').length).toBeGreaterThanOrEqual(1);
  });
});
```

---

## Integration Tests

### Advisory Board Component Tests

```javascript
// tests/integration/advisory-board.test.jsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AutoborosAdvisoryBoard } from '../../src/components/AutoborosAdvisoryBoard';

// Mock CopilotKit
vi.mock('@copilotkit/react-core', () => ({
  useCoAgent: () => ({ state: {}, setState: vi.fn(), run: vi.fn() }),
  useCopilotChat: () => ({ appendMessage: vi.fn(), isLoading: false }),
  useCopilotAction: vi.fn(),
}));

describe('AutoborosAdvisoryBoard', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('renders without crashing', () => {
    render(<AutoborosAdvisoryBoard />);
    expect(screen.getByText('Autoboros AI Advisory Board')).toBeTruthy();
  });

  it('shows all 7 advisor cards', () => {
    render(<AutoborosAdvisoryBoard />);
    expect(screen.getByText('PrimeNode Orchestrator')).toBeTruthy();
    expect(screen.getByText("Devil's Advocate")).toBeTruthy();
    expect(screen.getByText('Strategic Thinker')).toBeTruthy();
    expect(screen.getByText('Financial Analyst')).toBeTruthy();
    expect(screen.getByText('People Manager')).toBeTruthy();
    expect(screen.getByText('Innovation Catalyst')).toBeTruthy();
    expect(screen.getByText('Detail Tracker')).toBeTruthy();
  });

  it('toggles advisor selection on click', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<AutoborosAdvisoryBoard />);

    const devilsAdvocate = screen.getByText("Devil's Advocate").closest('div[class*="rounded-lg"]');
    await user.click(devilsAdvocate);

    // Verify visual active state applied
    expect(devilsAdvocate.className).toMatch(/border-red/);
  });

  it('disables Run Analysis button when no decision entered', () => {
    render(<AutoborosAdvisoryBoard />);
    const button = screen.getByRole('button', { name: /run analysis/i });
    expect(button).toBeDisabled();
  });

  it('enables Run Analysis button when decision is entered', async () => {
    const user = userEvent.setup();
    render(<AutoborosAdvisoryBoard />);

    const textarea = screen.getByPlaceholderText(/describe the decision/i);
    await user.type(textarea, 'Should we expand to Europe?');

    const button = screen.getByRole('button', { name: /run analysis/i });
    expect(button).not.toBeDisabled();
  });

  it('shows analysis results after running', async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<AutoborosAdvisoryBoard />);

    const textarea = screen.getByPlaceholderText(/describe the decision/i);
    await user.type(textarea, 'Should we launch a new product?');

    const button = screen.getByRole('button', { name: /run analysis/i });
    await user.click(button);

    // Advance past analysis timeout
    await vi.runAllTimersAsync();

    await waitFor(() => {
      expect(screen.getByText('Decision Analysis')).toBeTruthy();
    });
  });

  it('filters decisions by search term', async () => {
    // Set up a component with pre-existing decisions
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(<AutoborosAdvisoryBoard />);

    // Run two different analyses
    const textarea = screen.getByPlaceholderText(/describe the decision/i);
    await user.type(textarea, 'Should we hire more engineers?');
    await user.click(screen.getByRole('button', { name: /run analysis/i }));
    await vi.runAllTimersAsync();
    await waitFor(() => screen.getByText('Decision Analysis'));

    await user.clear(textarea);
    await user.type(textarea, 'What is our marketing budget?');
    await user.click(screen.getByRole('button', { name: /run analysis/i }));
    await vi.runAllTimersAsync();

    // Search for first decision
    const searchInput = screen.getByPlaceholderText(/search decisions/i);
    await user.type(searchInput, 'engineers');

    // Only matching decision should be visible
    expect(screen.queryByText(/marketing budget/i)).toBeNull();
  });
});
```

---

## E2E Tests

```javascript
// tests/e2e/decision-flow.test.js
import { test, expect } from '@playwright/test';

test.describe('Decision Analysis Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('complete decision analysis flow', async ({ page }) => {
    // Select advisors
    await page.click('text=Financial Analyst');
    await page.click("text=Devil's Advocate");

    // Verify advisors are selected
    await expect(page.locator('[data-advisor="financial-analyst"]')).toHaveClass(/border-green/);

    // Enter decision
    await page.fill('textarea', 'Should we invest $500k in AI infrastructure?');

    // Run analysis
    await page.click('button:has-text("Run Analysis")');

    // Wait for analysis to complete
    await page.waitForSelector('text=Decision Analysis', { timeout: 10000 });

    // Verify results appear
    await expect(page.locator('text=Financial Analyst')).toBeVisible();
    await expect(page.locator('text=confidence')).toBeVisible();
  });

  test('export decision as JSON', async ({ page }) => {
    // ... setup decision ...

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.click('button:has-text("Export JSON")')
    ]);

    const path = await download.path();
    expect(path).toBeTruthy();
    const content = JSON.parse(require('fs').readFileSync(path, 'utf-8'));
    expect(content).toHaveProperty('decision');
    expect(content).toHaveProperty('advisorInputs');
  });
});

// playwright.config.js
export default {
  testDir: './tests/e2e',
  use: {
    baseURL: 'http://localhost:5173',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
    { name: 'firefox', use: { browserName: 'firefox' } },
  ],
  webServer: {
    command: 'npm run dev',
    port: 5173,
    reuseExistingServer: !process.env.CI,
  },
};
```

---

## Test Coverage

### Coverage Targets

| Area | Target |
|------|--------|
| Unit (utils) | 90%+ |
| Integration (components) | 75%+ |
| E2E (critical flows) | Key user journeys |

### Generating Coverage Report

```bash
npm run test:coverage
# Opens coverage/index.html in browser
open coverage/index.html
```

### Coverage Configuration (vitest.config.js)

```javascript
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html', 'lcov'],
      include: ['src/**/*.{js,jsx,ts,tsx}'],
      exclude: ['src/main.jsx', 'src/**/*.d.ts'],
      thresholds: {
        lines: 75,
        functions: 75,
        branches: 70,
      }
    }
  }
});
```

---

## Mocking

### MSW Setup for API Mocking

```javascript
// tests/mocks/handlers.js
import { http, HttpResponse } from 'msw';

export const handlers = [
  http.post('/api/analyze', async ({ request }) => {
    const { decision, advisors } = await request.json();
    return HttpResponse.json({
      id: Date.now(),
      decision,
      advisorInputs: advisors.map(id => ({
        advisorId: id,
        confidence: 0.9,
        analysis: `Mock analysis for ${id}`
      }))
    });
  }),

  http.post('/api/export/pdf', () => {
    return new HttpResponse(new Blob(['mock-pdf-content']), {
      headers: { 'Content-Type': 'application/pdf' }
    });
  }),
];

// tests/setup.js
import { beforeAll, afterAll, afterEach } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from './mocks/handlers';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

---

## CI Testing

GitHub Actions runs the full suite on every push and PR. See `.github/workflows/deploy.yml` for configuration.

Tests run in this order:
1. Lint
2. Type check
3. Unit tests
4. Integration tests
5. Build verification
6. E2E tests (on staging deploy)

Failed tests block merging to main.
