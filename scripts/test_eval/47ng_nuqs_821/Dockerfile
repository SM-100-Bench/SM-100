FROM node:20

RUN npm install -g pnpm@latest

WORKDIR /repo

CMD ["bash", "-c", "cd packages/nuqs && pnpm install && pnpm build && pnpm test"]