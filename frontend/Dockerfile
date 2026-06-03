# Stage 1: Scaffolding and building Next.js static files
FROM node:20-alpine AS builder
WORKDIR /app

# Copy dependency files
COPY package*.json ./

# Install packages
RUN npm ci

# Copy full frontend workspace source code
COPY . .

# Compile and export NextJS as pure static files (under /app/out)
RUN npm run build

# Stage 2: Serve using high-performance lightweight Nginx
FROM nginx:alpine

# Copy static assets from build stage to Nginx directory
COPY --from=builder /app/out /usr/share/nginx/html

# Expose Nginx default port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
