const http = require("http");

const base = "10.252.134";
const requests = [];

for (let i = 0; i <= 255; i++) {    const host = `${base}.${i}`;

    requests.push(
      new Promise((resolve) => {
        const req = http.get(
          `http://${host}:11434/api/ps`,
          { timeout: 1000 },
          (res) => {
            res.resume();
            res.on("end", () => resolve(res.statusCode ? host : null));
          }
        );

        req.on("error", () => resolve(null));
        req.on("timeout", () => {
          req.destroy();
          resolve(null);
        });
      })
    );
  }

Promise.all(requests).then((results) => {
  results
    .filter(Boolean)
    .forEach((host) => console.log(`responded: ${host}`));
});