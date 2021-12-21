// `npm run doc`
import sn from "../main.js"
import fs from "fs"
fs.writeFileSync('./docs/README.md', sn.meta.docs())
console.log()
console.log("OK   Wrote docs to sn/js-lib/docs/README.md.")