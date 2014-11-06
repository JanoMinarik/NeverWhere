cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(ao-eval)

enable_language(CXX)

add_executable(
    ao-eval.x
    grid.cpp
    main.cpp
    neon.cpp
    )
