package com.springinaction.spittr.web

import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping

@Controller
@RequestMapping(*arrayOf("/", "/homepage"))
open class HomeController {

    @GetMapping
    open fun home(): String {
        return "home"
    }
}