// The export is `display(elem, 'Clicks per second', 15.6)`.
import 'https://d3js.org/d3.v6.min.js'



const sizes = {top: 10, right: 20, bottom: 20, left: 90, width: 450, height: 150}
const stepSizes = new WeakMap
const _minMaxBoundary = [null, false, `If checked, \`display\` will make plots span from min to max value in each pixel.
If unchecked, plots will span mean ± stddev in each pixel.`]



export default Object.assign(function display(into, lbl, vle, stepSize = 1) {
    if (lbl === undefined) {
        let L = into._display
        if (!(L instanceof Map)) return []
        return [...L.keys()].filter(k => k !== display)
    }
    if (vle === undefined) {
        // Remove the row.
        let L = into._display
        if (!(L instanceof Map)) return
        if (!L.has(lbl)) return
        L.get(lbl).parentNode.remove()
        L.delete(lbl)
    } else if (_isNumericArray(vle) || typeof vle == 'number' || vle === null || vle === 'empty') {
        let L = into._display
        if (!(L instanceof Map)) {
            L = new Map
            const tbl = elem('table')
            L.set(display, tbl)
            into._display = L
            into.append(tbl)

            const btn = elem('button', ['Copy JSON'])
            btn.addEventListener('click', () => {
                // Doesn't exactly fit under our "Browser support" table, but it's so much shorter than the non-modern way.
                const obj = {}
                L.forEach((v,k) => typeof k == 'string' && Array.isArray(v.to) && (obj[k] = v.to))
                navigator.clipboard.writeText(JSON.stringify(obj))
            })
            btn.style.width = '100%'
            tbl.append(elem('tr', [elem('td'), elem('td', [btn])]))
        }
        if (!_updatePlots.cells) _updatePlots.cells = new Set, _updatePlots.fn = _throttled(_updatePlots, .1)
        if (!L.has(lbl)) {
            // Create a table row with the label and the plot.
            const data = _isNumericArray(vle) ? [...vle] : typeof vle == 'number' ? [vle] : []
            const row = elem('tr', [elem('td', [lbl]), elem('td')])
            const cell = row.lastChild

            const dv = elem('div')
            const svg = d3.create('svg')
                .attr("width", sizes.width + sizes.left + sizes.right - .5) // (Firefox/Chromium agree only with this -.5.)
                .attr("height", sizes.height + sizes.top + sizes.bottom)
            const tooltip = elem('div')
            tooltip.style.position = 'absolute', tooltip.style.left = tooltip.style.top = 0, tooltip.style.pointerEvents = 'none'
            const num = elem('number')
            dv.append(svg.node(), num)
            num.textContent = data.length ? ''+data[0] : '<Nothing>',
            svg.node().style.display = 'none'

            cell.style.position = 'relative'
            cell.append(tooltip, dv)

            L.set(lbl, cell), cell.to = data
            stepSizes.set(data, stepSize || 1)
            L.get(display).append(row)

            cell.lastChild.firstChild.style.width = sizes.width + 'px'
            cell.lastChild.firstChild.style.height = sizes.height + 'px'

            if (typeof ResizeObserver != ''+void 0)
                (function(L, lbl, dv) {
                    new ResizeObserver(entries => {
                        L.has(lbl) && L.get(lbl).to.length > 1 && _updatePlotLater(L.get(lbl))
                    }).observe(dv)
                })(L, lbl, dv)
        } else if (_isNumericArray(vle))
            vle.forEach(v => L.get(lbl).to.push(v))
        else if (typeof vle == 'number')
            L.get(lbl).to.push(vle)
        else if (vle === 'empty')
            L.get(lbl).to.length = 0

        _updatePlotLater(L.get(lbl))
    } else
        error("Expected undefined or null or a number or a tensor, got", vle)
}, {
    docs:`\`display Into Label Value\` or \`display Into Label Value StepSize\`: displays a plot of all \`Value\`s at a \`Label\`. \`display Label\`: clears the display at a \`Label\`.    
The plot can display the exact values at cursor, and be zoomed in by a dragged click (and zoomed out by a quick click).

\`display(Into)\`: returns an array of all \`Label\`s.    
\`Value\` is \`undefined\`: delete the plot. \`null\`: initialize the plot if not initialized. \`'empty'\`: empty the plot (for overwriting).    
\`StepSize\` is the visual multiplier of each data index, which allows only giving data at regular indices.

(When zoomed out, high-variance lines are thicc.)`,
})



function elem(tag, content) {
    const el = document.createElement(tag)
    if (Array.isArray(content)) el.append(...content)
    return el
}
function _updatePlotLater(cell) {
    !_updatePlots.cells.size && setTimeout(_updatePlots.fn, 0)
    _updatePlots.cells.add(cell)
  
    const text = cell.firstChild.lastChild
    if (text && Array.isArray(cell.to)) _updatePlotTooltip(text, cell.to.length-1, cell.to, undefined, true)
}
function _updatePlotTooltip(text, x, data, makeEnd, exitIfNotEnd) {
    if (!text) return
    if (exitIfNotEnd && !text._isEnd) return
    const y = data[x], stepSize = stepSizes && stepSizes.get(data) || 1
    text.firstChild.textContent = 'At '+(x+1)*stepSize+', the value is\n'
    text.lastChild.textContent = (y < 1e8 ? y : (+y).toExponential(2))
    if (!exitIfNotEnd) text._isEnd = !!makeEnd
}
function _timeSince(t=0) { return performance.now() - t }
function _throttled(fun, cpu = .5, everyTime) {
    const blend = .3 // 1 to always estimate the next time as exactly the previous time.
    let lastDur = 0
    let scheduledTime = _timeSince(), scheduledId = null
    let lastRun = _timeSince()
    let arg1, arg2
    function throttled(x,y) {
        if (everyTime) everyTime(x,y)
        arg1 = x, arg2 = y
        let requiredRest = cpu === .5 ? lastDur : typeof cpu == 'number' ? lastDur * (1/cpu - 1) : cpu(lastDur)
        if (scheduledId) clearTimeout(scheduledId), requiredRest -= _timeSince(scheduledTime)
        else requiredRest -= Math.min(_timeSince(lastRun), 2000), lastRun = _timeSince()
        if (requiredRest > 2)
            scheduledId == null && (scheduledTime = _timeSince()), scheduledId = setTimeout(runThrottled, Math.max(0, requiredRest))
        else runThrottled(), scheduledTime = _timeSince()
    }
    function runThrottled() {
      scheduledId = null
      const start = _timeSince()
      const r = fun(arg1, arg2)
      arg1 = arg2 = undefined
      if (r instanceof Promise)
        r.then(userTimePassed => {
            if (typeof userTimePassed != 'number') userTimePassed = _timeSince(start)
            lastDur = blend * userTimePassed + (1-blend) * userTimePassed, lastRun = _timeSince()
        })
      else
        lastDur = blend * _timeSince(start) + (1-blend) * _timeSince(start), lastRun = _timeSince()
    }
    return throttled
}
function _updatePlots() {
    // Performs scheduled updates of plots.
    _updatePlots.cells.forEach(update)
  
    function update(cell) {
        _updatePlots.cells.delete(cell)
        if (typeof ResizeObserver != ''+void 0)
            cell.lastChild.classList.toggle('resizable', cell.to.length > 1)
        const hadText = cell.lastChild.lastChild.textContent
        if (cell.to.length > 1) {
            if (hadText)
                cell.lastChild.lastChild.textContent = '',
                cell.lastChild.firstChild.style.removeProperty('display')
            _updatePlot(d3.select(cell.lastChild.firstChild), sizeOf(cell.lastChild), cell.to)
        } else
            cell.lastChild.lastChild.textContent = cell.to.length ? ''+cell.to[cell.to.length-1] : '<Nothing>',
            cell.lastChild.firstChild.style.display = 'none'
    }
    function sizeOf(el) {
        if (el && el.offsetWidth && el.offsetWidth > 200) {
            const left = 90, bottom = 20
            return {top: 20, right: 20, bottom, left, width: el.offsetWidth - left - 20, height: el.offsetHeight - bottom - 20}
        }
        if (el && el.offsetWidth)
            return {top: 0, right: 0, bottom: 10, left: 20, width: el.offsetWidth - 20, height: el.offsetHeight - 10}
        return sizes
    }
    // .cells (a Set), .fn (a `_throttled` mirror of this function)
}
function _updatePlot(svg, sizes, data, begin, end) {
    if (!Array.isArray(data)) error("Expected an array, got", data)
    const el = svg.node()
    let transition = false
    if (begin === undefined)
        begin = el._begin !== undefined ? el._begin : 0
    else transition = true
    if (end === undefined)
        end = el._end !== undefined && el._end !== el._len ? el._end : data.length
    else transition = true
    const stepSize = stepSizes && stepSizes.get(data) || 1
  
    svg
        .attr("width", sizes.width + sizes.left + sizes.right - .5) // (Firefox/Chromium agree only with this -.5.)
        .attr("height", sizes.height + sizes.top + sizes.bottom)
    let xAxis, yAxis, plot
    if (!el.firstChild) {
        // If empty, create children, and attach events.
        xAxis = svg.append('g'), yAxis = svg.append('g'), plot = svg.append('path')
  
        const tooltip = svg.node().parentNode.previousSibling
        const focus = d3.select(elem('div')).style('opacity', 0).style('border', '.1em solid currentColor').style('transition', 'none')
        const text = d3.select(elem('text')).style('opacity', 0).style('color', 'currentColor').style('display', 'block').style('transition', 'none').style('text-align', 'center')
        focus.style('width', '.75em').style('height', '.75em').style('border-radius', '.5em')
        text.node().append(elem('text'), elem('number'))
        text.node().firstChild.style.textShadow = '-0.15em .15em .15em var(--background)'
        text.node().lastChild.style.textShadow = '-0.15em .15em .15em var(--background)'
        tooltip.append(focus.node(), text.node())
  
      // Also show the exact value at cursor.
        const zoom = svg.append('g').append('rect').style('fill', 'rgba(30,50,200,.3)').attr('y', 0).attr('height', '100%')
        let zoomBegin = null
        function mouseMove(evt) {
            let [cx,cy] = d3.pointer(evt, this)
            let i = Math.max(0, Math.min(Math.round(this._x.invert(cx)-1), this._len-1)), data = this._data
            if (i < 0 || i >= this._len)
                focus.style('opacity', 0), text.style('opacity', 0)
            else {
                const x = this._x(i+1), y = this._y(data[i])
                focus.style('opacity', 1), text.style('opacity', 1)
                focus.style('transform', `translate(-50%,-50%) translate(.3ch,0) translate(${x}px, ${y}px)`)
                _updatePlotTooltip(text.node(), i, data, i >= this._len-1)
                text.style('transform', `translate(-50%,-100%) translate(.3ch,0) translate(${x}px, ${y-20}px)`)
            }
  
            // Also display the rectangle of the future zoom.
            if (zoomBegin !== null) {
                let l = this._x(zoomBegin+1), r = this._x(i+1)
                if (zoomBegin === i) l = this._x(1), r = this._x(data.length)
                if (i >= 0 && i < this._len)
                    zoom.style('opacity', 1),
                    l<r ? zoom.attr('x', l).attr('width', r-l) : zoom.attr('x', r).attr('width', l-r)
                else zoom.style('opacity', 0)
            } else zoom.style('opacity', 0)
        }
        svg.on('pointermove', mouseMove)
            .on('pointerover', mouseMove)
            .on('pointerout',  () => { focus.style('opacity', 0), text.style('opacity', 0), zoom.style('opacity', 0) })
  
        // Also allow zooming in by a dragged click (and zooming out, by a quick click).
        svg.on('pointerdown', function(evt) {
            let [cx,cy] = d3.pointer(evt, this)
            const i = Math.max(0, Math.min(Math.round(this._x.invert(cx)-1), this._len-1))
            if (i >= 0 && i < this._len) zoomBegin = i
            evt.preventDefault()
            mouseMove.call(this, evt)
            svg.node().setPointerCapture && svg.node().setPointerCapture(evt.pointerId)
        }).on('pointerup', function(evt) {
            svg.node().releasePointerCapture && svg.node().releasePointerCapture(evt.pointerId)
            if (zoomBegin === null) return
            let [cx,cy] = d3.pointer(evt, this)
            let i = Math.max(0, Math.min(Math.round(this._x.invert(cx)-1), this._len-1)), data = this._data, sizes = this._sizes
            if (i >= 0 && i < this._len) {
                if (i === this._len-1) i = data.length-1
                if (zoomBegin === i) _updatePlot(d3.select(this), sizes, data, 0, data.length)
                else if (zoomBegin > i) _updatePlot(d3.select(this), sizes, data, i, zoomBegin+1)
                else if (zoomBegin < i) _updatePlot(d3.select(this), sizes, data, zoomBegin, i+1)
            }
            zoomBegin = null, zoom.style('opacity', 0)
            mouseMove.call(this, evt)
        })
  
    } else
        [xAxis, yAxis, plot] = el.childNodes, xAxis = d3.select(xAxis), yAxis = d3.select(yAxis), plot = d3.select(plot)
  
    const step = Math.max(1, ((end - begin) / sizes.width) | 0)
  
    // Zoom (also show a bit of values before the shown range, unless they're way out of range) (also show item distributions).
    let begin2 = begin
    const lookBehind = 0
    while (begin2 > 0 && begin2 > begin - lookBehind && data[begin2-1] >= Min - extra*10 && data[begin2-1] <= Max + extra*10) --begin2
    // Skip items, compute per-pixel(-ish) boundaries, trim offscreen points.
    const mins = new Array((end - begin2) / step | 0).fill(0), maxs = new Array((end - begin2) / step | 0).fill(0)
    if (_minMaxBoundary[1]) // Compute min/max for each pixel.
        for (let i=0; begin2 + i*step < end; ++i) {
            let a = begin2 + i*step, b = Math.min(begin2 + (i+1)*step, end), empty = true
            mins[i] = maxs[i] = 0
            for (let j = a; j < b; ++j)
                if (data[j] !== data[j] || !isFinite(data[j])) continue
                else if (empty) mins[i] = maxs[i] = data[j], empty = false
                else if (data[j] < mins[i]) mins[i] = data[j]
                else if (data[j] > maxs[i]) maxs[i] = data[j]
            if (empty && i) mins[i] = mins[i-1], maxs[i] = maxs[i-1]
        }
    else {
        // Compute the mean (in `mins`).
        for (let i=0; begin2 + i*step < end; ++i) {
            let a = begin2 + i*step, b = Math.min(begin2 + (i+1)*step, end)
            if (b-a < step && b-step > 0) a = b-step // Reduce last-pixel jumping.
            mins[i] = 0
            for (let j = a; j < b; ++j)
                if (data[j] !== data[j] || !isFinite(data[j])) continue
                else mins[i] += data[j]
            mins[i] /= b-a
        }
        // Compute stddev in each pixel (in `maxs`).
        for (let i=0; begin2 + i*step < end; ++i) {
            let a = begin2 + i*step, b = Math.min(begin2 + (i+1)*step, end)
            if (b-a < step && b-step > 0) a = b-step
            maxs[i] = 0
            for (let j = a; j < b; ++j)
                if (data[j] !== data[j] || !isFinite(data[j])) continue
                else maxs[i] += (data[j] - mins[i]) * (data[j] - mins[i])
            maxs[i] = Math.sqrt(maxs[i] / (b-a))
        }
        // Compute mean ± stddev (into `mins` and `maxs`).
        for (let i=0; begin2 + i*step < end; ++i) {
            const mn = mins[i], sd = maxs[i]
            mins[i] = mn - sd, maxs[i] = mn + sd
        }
    }
  
    // X axis.
    const Range = [sizes.left, sizes.left + sizes.width - sizes.right]
    const x = (el._x || (el._x = d3.scaleLinear()))
        .range(Range)
        .domain([begin+1, end])
    const x2 = (el._x2 || (el._x2 = d3.scaleLinear()))
        .range(Range)
        .domain([(begin+1) * stepSize, end * stepSize])
    ;(!transition ? xAxis : xAxis.transition(200))
        .attr("transform", `translate(0,${sizes.top + sizes.height - sizes.bottom})`)
        .call(d3.axisBottom(x2).ticks(Math.min(end - begin - 1, sizes.width / 80)).tickSizeOuter(0))
  
    // Y axis.
    let Min = mins[0], Max = maxs[0]
    for (let i = 1; i < end - begin; ++i) {
        if (mins[i] < Min) Min = mins[i]
        if (maxs[i] > Max) Max = maxs[i]
    }
    const extra = Math.abs(Max-Min)*0
    Min = (Min-extra<0) === (Min<0) ? Min-extra : 0, Max = (Max+extra<0) === (Max<0) ? Max+extra : 0
    const y = (el._y || (el._y = d3.scaleLinear()))
        .range([sizes.top + sizes.height - sizes.bottom, sizes.top])
        .domain([Min, Max]).nice()
    ;(!transition ? yAxis : yAxis.transition(200))
        .attr("transform", `translate(${sizes.left},0)`)
        .call(d3.axisLeft(y).ticks(sizes.height / 40).tickSizeOuter(0))
  
    el._x = x, el._y = y, el._data = data, el._sizes = sizes, el._begin = begin, el._end = end === data.length ? undefined : end, el._len = data.length, el._step = step
  
    // Plot.
    if (!plot.attr('fill'))
        plot.attr("fill", "steelblue")
            .attr("stroke", "steelblue")
            .attr("stroke-width", 1.5)
            .attr("stroke-linejoin", "round")
            .attr("stroke-linecap", "round")
    ;(!transition ? plot : plot.transition(200))
        .attr("d", "M " + mins.map((v,i) => 1+x(begin2 + i*step + 1) + " " + (y(v) || 0)).join(" L ") + " L " + maxs.reverse().map((v,i,maxs) => 1+x(begin2 + (maxs.length - i - 1)*step+1) + " " + (y(v) || 0)).join(" L ") + " Z")
}

function _isNumericArray(x) {
    return Array.isArray(x) && x.every(_isNum) || x instanceof Float32Array || x instanceof Float64Array || x instanceof Int32Array || x instanceof Int16Array || x instanceof Int8Array || x instanceof Uint32Array || x instanceof Uint16Array || x instanceof Uint8Array
}
function _isNum(x) { return typeof x == 'number' }

function error(...msg) { throw new Error(msg.join(' ')) }