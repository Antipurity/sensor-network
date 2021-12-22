// Importing this makes clicking on a `.hiding` DOM elem collapse `.hidable` direct child. Smoothly.
//   Can style `.hiding` based on `.isHiding`.
//   To pre-collapse, include `class="hiding isHiding"` on `.hiding` and `style="height: 0px"` on `.hidable`.
//   `.hidable` elems should have CSS `overflow: hidden;  transition: .3s` or similar.



addEventListener('click', evt => {
    let target = evt.target
    while (target && (!target.classList || !target.classList.contains('hiding') && !target.classList.contains('hidable')))
        target = target.parentNode
    if (!target || !target.classList || !target.classList.contains('hiding')) return
    let el = target.firstChild
    while (el && (!el.classList || !el.classList.contains('hidable'))) el = el.nextSibling
    if (!el || !el.classList || !el.classList.contains('hidable')) return
    const shown = !isHidden(el)
    if (shown) updateHeight(el)
    toggleHeight(el), el.style.setProperty('transition', 'none')
    if (!shown) updateHeight(el)
    toggleHeight(el), el.style.setProperty('transition', 'none')
    el.offsetHeight // Slow. But smooth.
    el.style.removeProperty('transition')
    toggleHeight(el)
    target.classList.toggle('isHiding', shown)
    function isHidden(el) {
        return el.style.height == '0px'
    }
    function toggleHeight(el) {
        el.style.height = isHidden(el) ? el._height + 'px' : '0px'
    }
    function updateHeight(el, isParent = false) {
        if (!el || !el.style || el === document.body) return
        updateHeight(el.parentNode, true)
        if (isParent && !el._height) return
        el.style.height = 'auto'
        el._height = el.offsetHeight
    }
}, {passive:true})
addEventListener('transitionend', evt => {
    const target = evt.target
    if (!target || !target.style) return
    if (target.style.height !== '0px' && target.style.height !== 'auto') target.style.removeProperty('height')
}, {passive:true})