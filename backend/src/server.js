import cors from 'cors';
import express from 'express';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import swaggerUi from 'swagger-ui-express';

import { PORT } from './config.js';
import logger from './logger.js';
import errorHandler from './middleware/errorHandler.js';
import healthRoutes from './routes/health.js';
import modelRoutes from './routes/model.js';
import pipelineRoutes from './routes/pipeline.js';
import predictRoutes from './routes/predict.js';
import sessionRoutes from './routes/session.js';
import wordRoutes from './routes/word.js';
import openapiSpec from './swagger/openapiSpec.js';
import { attachWebSocketServer } from './websocket/wsServer.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_DIR = path.resolve(__dirname, '..', '..', 'frontend');

export function createApp() {
  const app = express();
  app.use(cors());
  app.use(express.json({ limit: '5mb' }));

  app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(openapiSpec));
  app.get('/openapi.json', (req, res) => res.json(openapiSpec));

  app.use('/', healthRoutes);
  app.use('/api', sessionRoutes);
  app.use('/api', predictRoutes);
  app.use('/api', wordRoutes);
  app.use('/api', pipelineRoutes);
  app.use('/api', modelRoutes);

  // Serves frontend/index.html, style.css, app.js -- the UI and the API
  // are the same origin, so no CORS setup is needed on the frontend side.
  app.use(express.static(FRONTEND_DIR));

  app.use(errorHandler);
  return app;
}

export function startServer() {
  const app = createApp();
  const server = http.createServer(app);
  attachWebSocketServer(server);

  server.listen(PORT, () => {
    logger.info(`[SERVER] WordPredict Node backend listening on http://localhost:${PORT}`);
    logger.info(`[SERVER] Swagger UI: http://localhost:${PORT}/api-docs`);
    logger.info(`[SERVER] WebSocket:  ws://localhost:${PORT}/ws`);
  });
  return server;
}

if (process.argv[1] === __filename) {
  startServer();
}